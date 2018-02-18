##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable


def _unpack_modules(module_stack):
    params = []
    buffs = []
    for m in module_stack:
        params.extend(m.parameters())
        buffs.extend(m._all_buffers())
    return tuple(params), tuple(buffs)


def _to_cuda(x, device=None):
    x.data = x.data.cuda(device)
    if x._grad is not None:
        x._grad.data = x._grad.data.cuda(device)
    return x


def _possible_downsample(x, in_channels, out_channels, subsample=False,
                         use_gpu=False, device=None):
    out = x
    if subsample:
        out = F.avg_pool2d(out, 2, 2)
    if in_channels < out_channels:
        # Pad with empty channels
        pad = Variable(torch.zeros(out.size(0),
                                   (out_channels-in_channels)//2,
                                   out.size(2),
                                   out.size(3)),
                       requires_grad=True)
        if use_gpu:
            pad = _to_cuda(pad, device)
        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)
    return out


class _rev_block_function(Function):
    @staticmethod
    def residual(x, modules):
        """
        Compute a pre-activation residual function.

        Args:
            x (Variable) : The input variable.

        Returns:
            out (Variable) : The result of the computation.
        """
        out = x
        for m in modules:
            out = m(out)
        return out

    @staticmethod
    def _forward(x, in_channels, out_channels, f_modules, g_modules,
                 subsample=False, use_gpu=False, device=None):

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.no_grad():
            x1 = Variable(x1.contiguous())
            x2 = Variable(x2.contiguous())
            if use_gpu:
                x1 = _to_cuda(x1, device)
                x2 = _to_cuda(x2, device)

            x1_ = _possible_downsample(x1, in_channels, out_channels,
                                       subsample, use_gpu, device)
            x2_ = _possible_downsample(x2, in_channels, out_channels,
                                       subsample, use_gpu, device)

            # in_channels, out_channels
            f_x2 = _rev_block_function.residual(x2, f_modules)
            
            y1 = f_x2 + x1_
            
            # out_channels, out_channels
            g_y1 = _rev_block_function.residual(y1, g_modules)

            y2 = g_y1 + x2_
            
            y = torch.cat([y1, y2], dim=1)

            del y1, y2
            del x1, x2

        return y

    @staticmethod
    def _backward(output, f_modules, g_modules, use_gpu=False, device=None):

        y1, y2 = torch.chunk(output, 2, dim=1)
        with torch.no_grad():
            y1 = Variable(y1.contiguous())
            y2 = Variable(y2.contiguous())
            if use_gpu:
                y1 = _to_cuda(y1, device)
                y2 = _to_cuda(y2, device)

            # out_channels, out_channels
            x2 = y2 - _rev_block_function.residual(y1, g_modules)

            # in_channels, out_channels
            x1 = y1 - _rev_block_function.residual(x2, f_modules)

            del y1, y2
            x1, x2 = x1.data, x2.data

            x = torch.cat((x1, x2), 1)
        return x

    @staticmethod
    def _grad(x, dy, in_channels, out_channels, f_modules, g_modules,
              activations, subsample=False, use_gpu=False, device=None):
        dy1, dy2 = Variable.chunk(dy, 2, dim=1)

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.enable_grad():
            x1 = Variable(x1.contiguous(), requires_grad=True)
            x2 = Variable(x2.contiguous(), requires_grad=True)
            x1.retain_grad()
            x2.retain_grad()
            if use_gpu:
                x1 = _to_cuda(x1, device)
                x2 = _to_cuda(x2, device)

            x1_ = _possible_downsample(x1, in_channels, out_channels,
                                       subsample, use_gpu, device)
            x2_ = _possible_downsample(x2, in_channels, out_channels,
                                       subsample, use_gpu, device)

            # in_channels, out_channels
            f_x2 = _rev_block_function.residual(x2, f_modules)

            y1_ = f_x2 + x1_

            # in_channels, out_channels
            g_y1 = _rev_block_function.residual(y1_, g_modules)

            y2_ = g_y1 + x2_
            
            f_params, f_buffs = _unpack_modules(f_modules)
            g_params, g_buffs = _unpack_modules(g_modules)

            dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params), dy2,
                                      retain_graph=True)
            dy2_y1 = dd1[0]
            dgw = dd1[1:]
            dy1_plus = dy2_y1 + dy1
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params), dy1_plus,
                                      retain_graph=True)
            dfw = dd2[2:]

            dx2 = dd2[1]
            dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
            dx1 = dd2[0]

            activations.append(x)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_
            dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, f_modules, g_modules,
                activations, subsample=False, use_gpu=False, device=None,
                *args):
        """
        Compute forward pass including boilerplate code.

        This should not be called directly, use the apply method of this class.

        Args:
            ctx (Context) : Context object, see PyTorch docs.
            x (Tensor) : 4D input tensor.
            in_channels (int) : Number of channels on input.
            out_channels (int) : Number of channels on output.
            f_modules (List) : Sequence of modules for F function.
            g_modules (List) : Sequence of modules for G function.
            activations (List) : Activation stack.
            subsample (bool) : Whether to do 2x spatial pooling.
            use_gpu (bool) : Whether to use gpu.
            device (int) : GPU to use.
            *args: Should contain all the parameters of the module.
        """
        
        # if subsampling, information is lost and we need to save the input
        if subsample:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.f_modules = f_modules
        ctx.g_modules = g_modules
        ctx.subsample = subsample
        ctx.use_gpu = use_gpu
        ctx.device = device

        y = _rev_block_function._forward(x,
                                         in_channels,
                                         out_channels,
                                         f_modules,
                                         g_modules,
                                         subsample,
                                         use_gpu,
                                         device)

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            x = _rev_block_function._backward(output,
                                              ctx.f_modules,
                                              ctx.g_modules,
                                              ctx.use_gpu,
                                              ctx.device)

        dx, dfw, dgw = _rev_block_function._grad(x,
                                                 grad_out,
                                                 ctx.in_channels,
                                                 ctx.out_channels,
                                                 ctx.f_modules,
                                                 ctx.g_modules,
                                                 ctx.activations,
                                                 ctx.subsample,
                                                 ctx.use_gpu,
                                                 ctx.device)

        return (dx,) + (None,)*8 + tuple(dfw) + tuple(dgw)


class rev_block(nn.Module):
    """
    The base reversible block module. This implements a generic reversible
    block whose inputs can be recomputed from its outputs during backprop.
    The block can be composed out of any arbitrary stack of layers (modules),
    added via the add_module() method.
    
    To create custom blocks, it is recommended to subclass this class and
    add all necessary modules in __init__(). The forward() method must be left
    unmodified.
    
    Args:
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        activations (list) : A list must be created in the model that uses
            rev_block objects in order to store activations, as needed. Blocks
            such as this one pass around activations through this list.
        f_modules (list) : A list of modules implementing the f() path of a
            reversible block.
        g_modules (list) : A list of modules implemetning the g() path of a
            reversible block.
        subsample (bool) : Whether to perform 2x spatial subsampling.
        use_gpu (bool) : Whether to move compute and memory to GPU.
        device (int) : The GPU device ID to use if using GPU.

    Returns:
        out (Variable): The result of the computation
    """
    def __init__(self, in_channels, out_channels, activations,
                 f_modules=None, g_modules=None, subsample=False,
                 use_gpu=False, device=None):
        super(rev_block, self).__init__()
        # NOTE: channels are only counted for _possible_downsample()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activations = activations
        self.f_modules = f_modules if f_modules is not None else []
        self.g_modules = g_modules if g_modules is not None else []
        self.subsample = subsample
        self.use_gpu = use_gpu
        self.device = device
        
    def cuda(self, device=None):
        self.use_gpu = True
        self.device = device
        for m in self.f_modules:
            m.cuda(device)
        for m in self.g_modules:
            m.cuda(device)
        return self
        
    def cpu(self):
        self.use_gpu = False
        for m in self.f_modules:
            m.cpu()
        for m in self.g_modules:
            m.cpu()
        return self

    def add_module(self, module, in_channels=None, out_channels=None,
                   stride=None, **kwargs):
        '''
        Symmetrically stacks a module onto both the f() path and the g() path
        of the reversible block. The sequence of modules defines f() and g().
        
        Modules are instantiated anew for each path; keyword arguments are
        used to instantiate the module.  Only keyword arguments are allowed
        because the following arguments must be handled differently for
        f_modules and g_modules:
        
        in_channels, out_channels, stride
        
        Args:
            module (Module class) : An non-instantiated class specifiying a
                module to add to both the f() and g() paths.
            in_channels (int) : Use with modules that take this argument.
            out_channels (int) : Use with modules that take this argument.
            stride (int) : Use with modules that take this argument.
            **kwargs : Keyword arguments to instantiate the module.
        '''
        f_kwargs = dict(kwargs.items())
        g_kwargs = dict(kwargs.items())
        if in_channels is not None:
            f_kwargs['in_channels'] = in_channels
        if out_channels is not None:
            f_kwargs['out_channels'] = out_channels
            g_kwargs['in_channels'] = out_channels
            g_kwargs['out_channels'] = out_channels
        if stride is not None:
            f_kwargs['stride'] = stride
        self.f_modules.append(module(**f_kwargs))
        self.g_modules.append(module(**g_kwargs))
        self._register_module(self.f_modules[-1])
        self._register_module(self.g_modules[-1])
        
    def _register_module(self, module):
        i = 0
        try:
            name = module.__name__
        except AttributeError:
            name = module.__class__.__name__
        while '{}_{}'.format(name, i) in self._modules:
            i += 1
        name = '{}_{}'.format(name, i)
        super(rev_block, self).add_module(name=name, module=module)

    def forward(self, x):
        # Unpack parameters and buffers
        f_params, f_buffs = _unpack_modules(self.f_modules)
        g_params, g_buffs = _unpack_modules(self.g_modules)
        
        return _rev_block_function.apply(x,
                                         self.in_channels//2,
                                         self.out_channels//2,
                                         self.f_modules,
                                         self.g_modules,
                                         self.activations,
                                         self.subsample,
                                         self.use_gpu,
                                         self.device,
                                         *f_params,
                                         *g_params)


class batch_normalization(nn.Module):
    """
    A wrapper for BatchNorm that can be used in rev_block. Since in_channels
    needs to be sometimes different on the f() and g() paths of a rev_block,
    it is useful to specify in_channels and out_channels arguments when
    adding a normalization module to a rev_block.
    
    Args:
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        ndim (int) : The number of spatial dimensions (1, 2 or 3).
        *args : Passed to BatchNorm.
        **kwargs : Passed to BatchNorm.
    
    Returns:
        out (Variable): The result of the computation.
    """
    def __init__(self, in_channels, out_channels, ndim=2, *args, **kwargs):
        super(batch_normalization, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        if ndim==1:
            self.norm_op = nn.BatchNorm1d
        elif ndim==2:
            self.norm_op = nn.BatchNorm2d
        elif ndim==3:
            self.norm_op = nn.BatchNorm3d
        else:
            raise ValueError("ndim must be 1, 2 or 3")
        if 'num_features' in kwargs:
            raise TypeError("must use \'in_channels\' instead of "
                            "\'num_features\'")
        self.norm = self.norm_op(*args,
                                 num_features=in_channels,
                                 **kwargs)
        
    def forward(self, x):
        return self.norm(x)
    

class basic_block(rev_block):
    """
    Implements the standard ResNet "basic block" as a reversible block.
    
    Args:
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        activations (list) : List to track activations, as in rev_block.
        subsample (bool) : Whether to perform 2x spatial subsampling.
        
    Returns:
        out (Variable): The result of the computation.
    """
    def __init__(self, in_channels, out_channels, activations,
                 subsample=False):
        super(basic_block, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          activations=activations,
                                          subsample=subsample)
        self.add_module(batch_normalization,
                        in_channels=in_channels//2,
                        out_channels=out_channels//2)
        self.add_module(nn.ReLU)
        self.add_module(nn.Conv2d,
                        in_channels=in_channels//2,
                        out_channels=out_channels//2,
                        kernel_size=3,
                        padding=1,
                        stride=2 if subsample else 1)
        self.add_module(batch_normalization,
                        in_channels=out_channels//2,
                        out_channels=out_channels//2)
        self.add_module(nn.ReLU)
        self.add_module(nn.Conv2d,
                        in_channels=out_channels//2,
                        out_channels=out_channels//2,
                        kernel_size=3,
                        padding=1)
