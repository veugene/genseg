##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

from inplace_abn.bn import ABN
from fcn_maker.blocks import (get_initializer,
                              convolution,
                              max_pooling,
                              do_upsample,
                              batch_normalization,
                              block_abstract,
                              shortcut)
from .modules import overlap_tile


"""
Return AlphaDropout if nonlinearity is 'SELU', else Dropout.
"""
def get_dropout(nonlin=None):
    if nonlin=='SELU':
        return torch.nn.AlphaDropout
    return torch.nn.Dropout


"""
Return a nonlinearity from the core library or return the provided function.
"""
def get_nonlinearity(nonlin):
    if nonlin is None:
        class identity_activation(torch.nn.Module):
            def forward(self, input):
                return input
        return identity_activation
        
    # Identify function.
    func = None
    if isinstance(nonlin, str):
        # Find the nonlinearity by name.
        try:
            func = getattr(torch.nn.modules.activation, nonlin)
        except AttributeError:
            raise ValueError("Specified nonlinearity ({}) not found."
                             "".format(nonlin))
    else:
        # Not a name; assume a module is passed instead.
        func = nonlin
        
    return func


def _unpack_modules(module_stack):
    params = []
    buffs = []
    for m in module_stack:
        params.extend(m.parameters())
        buffs.extend(m._all_buffers())
    return tuple(params), tuple(buffs)


def _to_cuda(x, device=None):
    if not x.data.is_cuda:
        x.data = x.data.cuda(device)
    if x._grad is not None and not x._grad.is_cuda:
        x._grad.data = x._grad.data.cuda(device)
    return x


def _adjust_tensor_size(x, in_channels, out_channels, subsample=False,
                        ndim=2):
    out = x
    if subsample:
        assert ndim in (1,2,3)
        if ndim==1:
            out = F.avg_pool1d(out, 2)
        elif ndim==2:
            out = F.avg_pool2d(out, 2, 2)
        else:
            out = F.avg_pool3d(out, 2, 2, 2)
    
    if in_channels < out_channels:
        # Pad with empty channels.
        padded_size = list(out.size())
        padded_size[1] = (out_channels-in_channels)//2
        pad = Variable(torch.zeros(padded_size), requires_grad=x.requires_grad)
        if x.is_cuda:
            pad = _to_cuda(pad, x.get_device())
        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)
        
    return out


def _unpad(x, in_channels, out_channels):
    # Extract half the channels, assuming the tensor was padded.
    # (backward pass)
    out = x
    if in_channels < out_channels:
        pad_channels = (out_channels-in_channels)//2
        out = x[:,pad_channels:-pad_channels]
        out.data = out.data.contiguous()
        if x.is_cuda:
            out = _to_cuda(out, out.get_device())
    return out


def _split(x):
    return x[:,:x.size(1)//2], x[:,x.size(1)//2:]


class _rev_block_function(Function):
    @staticmethod
    def _apply_modules(x, modules):
        out = x
        out.data = out.data.contiguous()
        for m in modules:
            out = m(out)
        return out

    @staticmethod
    def _forward(x, in_channels, out_channels, f_modules, g_modules,
                 subsample=False, ndim=2):
        
        x1, x2 = _split(x)
        with torch.no_grad():
            x1_ = _adjust_tensor_size(x1, in_channels, out_channels,
                                      subsample, ndim)
            x2_ = _adjust_tensor_size(x2, in_channels, out_channels,
                                      subsample, ndim)

            # in_channels, out_channels
            f_x2 = _rev_block_function._apply_modules(x2, f_modules)
            
            y1 = f_x2 + x1_
            
            # out_channels, out_channels
            g_y1 = _rev_block_function._apply_modules(y1, g_modules)

            y2 = g_y1 + x2_
            
            y = torch.cat([y1, y2], dim=1)

        del y1, y2
        del x1, x2

        return y

    @staticmethod
    def _backward(output, in_channels, out_channels, f_modules, g_modules):

        y1, y2 = _split(output)
            
        with torch.no_grad():
            # out_channels, out_channels
            x2_ = y2 - _rev_block_function._apply_modules(y1, g_modules)   
            x2 = _unpad(x2_, in_channels, out_channels)

            # in_channels, out_channels
            x1_ = y1 - _rev_block_function._apply_modules(x2, f_modules)
            x1 = _unpad(x1_, in_channels, out_channels)

        del y1, y2

        x = torch.cat((x1, x2), 1)
        
        return x

    @staticmethod
    def _grad(x, dy, in_channels, out_channels, f_modules, g_modules,
              backprop_buffer, subsample=False, ndim=2):
        dy1, dy2 = _split(dy)
        x1, x2 = _split(x)
        
        x1.data = x1.data.contiguous()
        x2.data = x2.data.contiguous()
        x1.requires_grad = True
        x2.requires_grad = True
        if x.is_cuda:
            x1 = _to_cuda(x1, x.get_device())
            x2 = _to_cuda(x2, x.get_device())

        with torch.enable_grad():
            x1_ = _adjust_tensor_size(x1, in_channels, out_channels,
                                      subsample, ndim)
            x2_ = _adjust_tensor_size(x2, in_channels, out_channels,
                                      subsample, ndim)

            # in_channels, out_channels
            f_x2 = _rev_block_function._apply_modules(x2, f_modules,)

            y1_ = f_x2 + x1_

            # in_channels, out_channels
            g_y1 = _rev_block_function._apply_modules(y1_, g_modules)

            y2_ = g_y1 + x2_
            
            f_params, f_buffs = _unpack_modules(f_modules)
            g_params, g_buffs = _unpack_modules(g_modules)

            dd1 = torch.autograd.grad(y2_, (y1_,)+tuple(g_params), dy2)
            dy2_y1 = dd1[0]
            dgw = dd1[1:]
            dy1_plus = dy2_y1 + dy1
            dd2 = torch.autograd.grad(y1_, (x1, x2)+tuple(f_params), dy1_plus)
            dfw = dd2[2:]

            dx2 = dd2[1]
            dx2 += torch.autograd.grad(x2_, x2, dy2)[0]
            dx1 = dd2[0]

        backprop_buffer['activations'].append(x)

        y1_.detach_()
        y2_.detach_()
        del y1_, y2_
        dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, f_modules, g_modules,
                backprop_buffer, subsample=False, ndim=2, *args):
        """
        Compute forward pass including boilerplate code.

        This should not be called directly, use the apply method of this class.

        Args:
            ctx (Context) : Context object, see PyTorch docs.
            x (Variable) : 4D input Variable.
            in_channels (int) : Number of channels on input.
            out_channels (int) : Number of channels on output.
            f_modules (List) : Sequence of modules for F function.
            g_modules (List) : Sequence of modules for G function.
            backprop_buffer (dict) : Must contain 'activations' list and
                'forward_passes' list for memory book-keeping.
            subsample (bool) : Whether to do 2x spatial pooling.
            ndim (int) : The number of spatial dimensions (1, 2 or 3).
            *args: Should contain all the parameters of the module.
        """
        
        # if subsampling, information is lost and we need to save the input
        if subsample:
            backprop_buffer['activations'].append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.backprop_buffer = backprop_buffer
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.f_modules = f_modules
        ctx.g_modules = g_modules
        ctx.subsample = subsample
        ctx.ndim = ndim

        y = _rev_block_function._forward(x,
                                         in_channels,
                                         out_channels,
                                         f_modules,
                                         g_modules,
                                         subsample,
                                         ndim)
        
        backprop_buffer['forward_passes'][-1] += 1

        return y

    @staticmethod
    def backward(ctx, grad_out):
        # Load or reconstruct input
        if ctx.load_input:
            ctx.backprop_buffer['activations'].pop()
            x = ctx.backprop_buffer['activations'].pop()
        else:
            output = ctx.backprop_buffer['activations'].pop()
            x = _rev_block_function._backward(output,
                                              ctx.in_channels,
                                              ctx.out_channels,
                                              ctx.f_modules,
                                              ctx.g_modules)

        dx, dfw, dgw = _rev_block_function._grad(x,
                                                 grad_out,
                                                 ctx.in_channels,
                                                 ctx.out_channels,
                                                 ctx.f_modules,
                                                 ctx.g_modules,
                                                 ctx.backprop_buffer,
                                                 ctx.subsample,
                                                 ctx.ndim)
        
        ctx.backprop_buffer['forward_passes'][-1] -= 1
        if ctx.backprop_buffer['forward_passes'][-1] == 0:
            # No more backprop in this chain.
            ctx.backprop_buffer['forward_passes'].pop()
            ctx.backprop_buffer['activations'].pop()

        return (dx,) + (None,)*7 + tuple(dfw) + tuple(dgw)


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
        backprop_buffer (dict) : Must contain 'activations' list and
            'forward_passes' list for memory book-keeping.
        f_modules (list) : A list of modules implementing the f() path of a
            reversible block.
        g_modules (list) : A list of modules implemetning the g() path of a
            reversible block.
        subsample (bool) : Whether to perform 2x spatial subsampling.
        ndim (int) : The number of spatial dimensions (1, 2 or 3).

    Returns:
        out (Variable): The result of the computation
    """
    def __init__(self, in_channels, out_channels, backprop_buffer,
                 f_modules=None, g_modules=None, subsample=False,
                 ndim=2):
        super(rev_block, self).__init__()
        # NOTE: channels are only counted for _adjust_tensor_size()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backprop_buffer = backprop_buffer
        self.f_modules = nn.ModuleList(f_modules)
        self.g_modules = nn.ModuleList(g_modules)
        self.subsample = subsample
        self.ndim = ndim

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
                                         self.backprop_buffer,
                                         self.subsample,
                                         self.ndim,
                                         *f_params,
                                         *g_params)
    
    
class _make_recompute_on_backward(Function):
    @staticmethod
    def forward(ctx, x, module_list, *params):
        ctx.save_for_backward(x, *params)
        ctx.module_list = module_list
        out = x
        with torch.no_grad():
            for m in module_list:
                out = m(out)
        return out
        
    def backward(ctx, grad_out):
        params, buffs = _unpack_modules(ctx.module_list)
        x = ctx.saved_variables[0]
        out = Variable(x.data, requires_grad=True)
        with torch.enable_grad():
            for i, m in enumerate(ctx.module_list):
                out = m(out)
        if x.requires_grad:
            grads = torch.autograd.grad(out, (x,)+tuple(params), grad_out)
            grad_x = grads[0]
            grad_params = grads[1:]
        else:
            grads = torch.autograd.grad(out, tuple(params), grad_out)
            grad_x = None
            grad_params = grads
        return (grad_x, None)+tuple(grad_params)
    
    
class make_recompute_on_backward(nn.Module):
    def __init__(self, module_list):
        super(make_recompute_on_backward, self).__init__()
        self.module_list = nn.ModuleList(module_list)
        
    def forward(self, x):
        params, buffs = _unpack_modules(self.module_list)
        return _make_recompute_on_backward.apply(x, self.module_list, *params)
    
    
class recomputable_norm_nonlin(nn.Module):
    def __init__(self, nonlinearity='ReLU', nonlin_kwargs=None,
                 norm_callable=batch_normalization, **bn_kwargs):
        super(bn_nonlin, self).__init__()
        bn = bn_callable(**bn_kwargs)
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        nonlin = get_nonlinearity(nonlinearity)(**nonlin_kwargs)
        self.op = recompute_on_backward([bn, nonlin])
        
    def forward(self, x):
        return self.op(x)


class rn_batch_normalization(nn.Module):
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
        super(rn_batch_normalization, self).__init__()
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
    
    
class activated_batch_normalization(nn.Module):
    """
    A wrapper for ABN that can be used in rev_block. Since in_channels
    needs to be sometimes different on the f() and g() paths of a rev_block,
    it is useful to specify in_channels and out_channels arguments when
    adding a normalization module to a rev_block.
    
    Args:
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        *args : Passed to BatchNorm.
        **kwargs : Passed to BatchNorm.
    
    Returns:
        out (Variable): The result of the computation.
    """
    def __init__(self, in_channels, out_channels, *args,
                 recompute_on_backward=False, **kwargs):
        super(activated_batch_normalization, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.recompute_on_backward = recompute_on_backward
        if 'num_features' in kwargs:
            raise TypeError("must use \'in_channels\' instead of "
                            "\'num_features\'")
        norm = ABN(*args,
                   num_features=in_channels,
                   **kwargs)
        if recompute_on_backward:
            norm = make_recompute_on_backward([norm])
        self.norm = norm
        
    def forward(self, x):
        return self.norm(x)
    

class reversible_basic_block(rev_block):
    """
    Implements the standard ResNet "basic block" as a reversible block.
    
    Args:
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        backprop_buffer (dict) : Must contain 'activations' list and
                'forward_passes' list for memory book-keeping.
        subsample (bool) : Whether to perform 2x spatial subsampling.
        dilation (int) : The dilation of the first convolution.
        dropout (float) : Dropout probability.
        norm_kwargs (dict): Keyword arguments to pass to batch norm layers.
        init (string or function) : Convolutional kernel initializer.
        nonlinearity (string or function) : Nonlinearity.
        ndim (int) : Number of spatial dimensions (1, 2 or 3).
        
    Returns:
        out (Variable): The result of the computation.
    """
    def __init__(self, in_channels, out_channels, backprop_buffer,
                 subsample=False, dilation=1, dropout=0., norm_kwargs=None, 
                 init='kaiming_normal', nonlinearity='ReLU', ndim=2):
        super(reversible_basic_block, self).__init__( \
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            backprop_buffer=backprop_buffer,
                                            subsample=subsample)
        self.dilation = dilation
        self.dropout = dropout
        self.norm_kwargs = norm_kwargs
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        # Build block.
        bn_nonlin = get_nonlinearity(nonlinearity)
        if bn_nonlin is not None:
            bn_nonlin = bn_nonlin()
        self.add_module(activated_batch_normalization,
                        in_channels=in_channels//2,
                        out_channels=out_channels//2,
                        activation=bn_nonlin,
                        **norm_kwargs)
        self.add_module(convolution,
                        ndim=ndim,
                        init=init,
                        kernel_size=3,
                        padding=dilation,
                        stride=2 if subsample else 1,
                        dilation=dilation,
                        in_channels=in_channels//2,
                        out_channels=out_channels//2)
        if dropout > 0:
            self.add_module(get_dropout(nonlinearity),
                            p=dropout)
        self.add_module(activated_batch_normalization,
                        in_channels=out_channels//2,
                        out_channels=out_channels//2,
                        activation=bn_nonlin,
                        **norm_kwargs)
        self.add_module(convolution,
                        ndim=ndim,
                        init=init,
                        kernel_size=3,
                        padding=1,
                        in_channels=out_channels//2,
                        out_channels=out_channels//2)


class dilated_rev_block(block_abstract):
    def __init__(self, in_channels, num_blocks, num_filters, dilation,
                 subsample=False, upsample=False, upsample_mode='repeat',
                 dropout=0., norm_kwargs=None, init='kaiming_normal',
                 nonlinearity='ReLU', block_type=reversible_basic_block,
                 ndim=2):
        """
        Implements a reversible dilated fully convolutional sequence of 
        blocks.
        
        Args:
            in_channels (int) : The number of input channels.
            num_blocks (int) : The number of blocks to stack.
            num_filters (list or int) : Number of convolution filters per
                block.
            dilation (list or int) : Convolutional kernel dilation per block.
            subsample (bool) : Subsample at input.
            upsample (bool) : Upsample at output.
            upsample_mode (string) : Either 'conv' or 'repeat'. If 'conv',
                does transposed convolution. If repeat, repeats rows and
                columns (nearest neighbour interpolation).
            dropout (float) : Probability of dropout.
            norm_kwargs (dict): Keyword arguments to pass to batch norm layers.
            init (string or function) : Convolutional kernel initializer.
            nonlinearity (string or function) : Nonlinearity.
            block_type (Module) : The block type to use.
            ndim : Number of spatial dimensions (1, 2 or 3).
        """
        super(dilated_rev_block, self).__init__(in_channels, num_filters,
                                                subsample, upsample)
        self._backprop_buffer = {'activations': [],
                                 'forward_passes': []}
        self.out_channels = num_filters[-1]
        self.num_blocks = num_blocks
        if hasattr(num_filters, '__len__'):
            self.num_filters = [i for i in num_filters]
        else:
            self.num_filters = [num_filters]*num_blocks
        if hasattr(dilation, '__len__'):
            self.dilation = [i for i in dilation]
        else:
            self.dilation = [dilation]*num_blocks
        self.upsample_mode = upsample_mode
        self.dropout = dropout
        self.norm_kwargs = norm_kwargs
        self.init = init
        self.nonlinearity = nonlinearity
        self.block_type = block_type
        self.ndim = ndim
        assert len(num_filters)==num_blocks
        assert len(dilation)==num_blocks
        
        if subsample:
            self.subsample_op = max_pooling(kernel_size=2, ndim=ndim)
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        for i in range(num_blocks):
            block = block_type(in_channels=prev_channels,
                               out_channels=num_filters[i],
                               nonlinearity=nonlinearity,
                               dropout=dropout,
                               norm_kwargs=norm_kwargs,
                               dilation=dilation[i],
                               init=init,
                               ndim=ndim,
                               backprop_buffer=self._backprop_buffer)
            self.layers.append(block)
            prev_channels = num_filters[i]
        if upsample:
            self.upsample_op = do_upsample(mode=upsample_mode,
                                           ndim=ndim,
                                           in_channels=num_filters[-1],
                                           out_channels=num_filters[-1],
                                           kernel_size=2,
                                           init=init)
            
    def forward(self, x):
        if self.subsample:
            x = self.subsample_op(x)
        self._backprop_buffer['forward_passes'].append(0)
        for layer in self.layers:
            x = layer(x)
        self._backprop_buffer['activations'].append(x)
        if self.upsample:
            x = self.upsample_op(x)
        return x    
    
    
class tiny_block(block_abstract):
    def __init__(self, in_channels, num_filters, input_patch_size=None,
                 subsample=False, upsample=False, upsample_mode='repeat',
                 skip=True, dropout=0., norm_kwargs=None,
                 init='kaiming_normal', nonlinearity='ReLU', ndim=2):
        super(tiny_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.input_patch_size = input_patch_size
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.norm_kwargs = norm_kwargs
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        bn_nonlin = get_nonlinearity(nonlinearity)
        if bn_nonlin is not None:
            bn_nonlin = bn_nonlin()
        self.op += [activated_batch_normalization(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  activation=bn_nonlin,
                                                  **norm_kwargs)]
        if subsample:
            self.op += [max_pooling(kernel_size=2, ndim=ndim)]
        padding = 0 if self.input_patch_size is not None else 1
        conv = convolution(in_channels=in_channels,
                           out_channels=num_filters,
                           kernel_size=3, 
                           ndim=ndim,
                           init=init,
                           padding=padding)
        if self.input_patch_size is not None:
            self.op += [overlap_tile(input_patch_size,
                                     model=conv,
                                     in_channels=conv.in_channels,
                                     out_channels=conv.out_channels)]
        else:
            self.op += [conv]
        if dropout > 0:
            self.op += [get_dropout(nonlinearity)(dropout)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=num_filters,
                                    out_channels=num_filters,
                                    kernel_size=2,
                                    init=init)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out
    

def memoryless_block_wrapper(block_list):
    def f(**kwargs):
        # Copy keyword arguments and extract `in_channels` (if it exists) as
        # settings for the first block. Remove any `subsample` or `upsample`
        # arguments since these must be passed per block, instead.
        kwargs_ = dict(kwargs.items())
        in_channels = None
        out_channels = None
        if 'in_channels' in kwargs_:
            in_channels = kwargs_.pop('in_channels')
        if 'subsample' in kwargs_:
            kwargs_.pop('subsample')
        if 'upsample' in kwargs_:
            kwargs_.pop('upsample')
        
        # Create every block.
        block_obj_list = []
        for block_type, block_kwargs in block_list:
            # Copy keyword arguments specific to this block.
            block_kwargs_ = dict(block_kwargs.items())
            
            # Merge in general keyword arguments.
            block_kwargs.update(kwargs_)
            
            # Set `in_channels` according to the arguments specific to this
            # block or according to the previous block's output (or the
            # initial `in_channels` setting, in that order.
            if 'in_channels' in block_kwargs_:
                in_channels_ = block_kwargs['in_channels']
            elif len(block_obj_list) > 0:
                in_channels_ = block_obj_list[-1].out_channels
            else:
                in_channels_ = in_channels
                
            # Instantiate the block.
            block = block_type(in_channels=in_channels_, **block_kwargs_)
            block_obj_list.append(block)
            
        # Wrap block list, forcing pass through all blocks on backprop.
        block_wrapped = make_recompute_on_backward(block_obj_list)
        
        # Record `in_channels`, and `out_channels` attributes for the
        # resulting object.
        block_wrapped.in_channels = block_obj_list[0].in_channels
        block_wrapped.out_channels = block_obj_list[-1].out_channels
        return block_wrapped
    return f
