import torch
from torch import nn
import numpy as np
from fcn_maker.blocks import (get_nonlinearity,
                              get_initializer,
                              do_upsample,
                              convolution,
                              block_abstract,
                              identity_block,
                              basic_block,
                              tiny_block,
                              dense_block,
                              repeat_block,
                              adjust_to_size)

 
def get_output_shape(layer, input_shape):
    """
    Works for `convolution`, `nn.Linear`, `identity_block`, `basic_block`,
    `tiny_block`, `dense_block`, `repeat_block`.
    
    `input_shape` is without batch dimension.
    """
    def _padding_array(padding):
        if not hasattr(padding, '__len__'):
            return np.array(2*padding)
        arr = [padding[i]+padding[i+1] for i in range(0, len(padding), 2)]
        return np.array(arr)
        
    def compute_conv_out_shape(input_shape, out_channels, padding,
                               kernel_size, stride=1):
        out_shape = 1 + ( np.array(input_shape)[1:]
                         +_padding_array(padding)
                         -np.array(kernel_size)) // np.array(stride)
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_tconv_out_shape(input_shape, out_channels, padding,
                                kernel_size, stride=1):
        out_shape = ( (np.array(input_shape[1:])-1)*np.array(stride)
                     -_padding_array(padding)
                     +np.array(kernel_size))
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_pool_out_shape(input_shape, padding,
                               stride=2, ceil_mode=False):
        input_spatial_shape_padded = ( np.array(input_shape)[1:]
                                      +_padding_array(padding))
        out_shape = input_spatial_shape_padded//stride
        if ceil_mode:
            out_shape += input_spatial_shape_padded%stride
        out_shape = (input_shape[0],)+tuple(out_shape)
        return out_shape
    
    def compute_block_upsample(layer, input_shape, kernel_size=3, stride=1):
        if layer.upsample_mode=='conv':
            out_shape = compute_tconv_out_shape(
                                           input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=0,
                                           kernel_size=kernel_size,
                                           stride=stride)
        elif layer.upsample_mode=='repeat':
            out_shape = (input_shape[0],)+tuple(np.array(input_shape[1:])*2)
        else:
            raise AssertionError("Invalid `upsample_mode`: {}"
                                 "".format(layer.upsample_mode))
        return out_shape
    
    if isinstance(layer, convolution):
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=layer.padding,
                                           kernel_size=layer.op.kernel_size,
                                           stride=layer.op.stride)
        return out_shape
    elif isinstance(layer, nn.Linear):
        return (layer.out_features,)
    elif isinstance(layer, identity_block):
        return input_shape
    elif isinstance(layer, basic_block):
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=2 if layer.subsample else 1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=1)
        return out_shape
    elif isinstance(layer, tiny_block):
        out_shape = input_shape
        if layer.subsample:
            out_shape = compute_pool_out_shape(input_shape=input_shape,
                                               padding=0,
                                               stride=2)
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        return out_shape
    elif isinstance(layer, conv_block):
        padding = 0
        if layer.conv_padding:
            padding = [(layer.kernel_size-1)//2,
                       (layer.kernel_size-int(layer.subsample))//2]*layer.ndim
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=padding,
                                           kernel_size=layer.kernel_size,
                                           stride=2 if layer.subsample else 1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        return out_shape
    elif isinstance(layer, dense_block):
        raise NotImplementedError("TODO: implement shape inference for "
                                  "`dense_block`.")
    elif isinstance(layer, repeat_block):
        out_shape = input_shape
        for block in layer.blocks:
            out_shape = get_output_shape(block, out_shape)
        return out_shape
    else:
        raise NotImplementedError("Shape inference not implemented for "
                                  "layer type {}.".format(type(layer)))
    

def batch_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.BatchNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.BatchNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.BatchNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 1, 2, or 3")
    

def instance_normalization(ndim=2, *args, **kwargs):
    if ndim==1:
        return torch.nn.InstanceNorm1d(*args, **kwargs)
    if ndim==2:
        return torch.nn.InstanceNorm2d(*args, **kwargs)
    elif ndim==3:
        return torch.nn.InstanceNorm3d(*args, **kwargs)
    else:
        raise ValueError("ndim must be 1, 2, or 3")


def group_normalization(num_features, ndim=None, *args, **kwargs):
    return torch.nn.GroupNorm(*args, num_channels=num_features, **kwargs)


class layer_normalization(nn.Module):
    """
    Ming-Yu's layer normalization implementation, to avoid the ridiculous need
    to specify output shape with pytorch's default LayerNorm implementation.
    """
    def __init__(self, num_features, eps=1e-5, affine=True, ndim=None):
        super(layer_normalization, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two 
            # lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class norm_nlin_fc(torch.nn.Module):
    """
    Helper to build a norm -> ReLU -> fully-connected.
    """
    def __init__(self, in_features, out_features, nonlinearity='ReLU',
                 normalization=instance_normalization, norm_kwargs=None,
                 init='kaiming_normal_'):
        super(norm_nlin_fc, self).__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_features = in_features
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.init = init
        if normalization is not None:
            self._modules['norm'] = normalization(
                                                ndim=1,
                                                num_features=self.in_features,
                                                **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        self._modules['fc'] = nn.Linear(in_features=self.in_features,
                                        out_features=self.out_features)
        if init is not None:
            self._modules['fc'].weight.data = \
                get_initializer(init)(self._modules['fc'].weight.data)

    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(encoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels = input_shape[0]
        self.out_channels = self.num_channels_list[-1]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = self.in_channels
        conv = convolution(in_channels=last_channels,
                           out_channels=self.num_channels_list[0],
                           kernel_size=7,
                           stride=1,
                           padding=3,
                           padding_mode=self.padding_mode,
                           init=self.init)
        self.blocks.append(conv)
        shape = get_output_shape(conv, shape)
        last_channels = self.num_channels_list[0]
        for i in range(1, self.num_conv_blocks):
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[i],
                                    subsample=True,
                                    skip=skip,
                                    dropout=self.dropout,
                                    normalization=self.normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[i]
        if normalization is not None:
            block = normalization(ndim=self.ndim,
                                  num_features=last_channels,
                                  **self.norm_kwargs)
            self.blocks.append(block)
        self.blocks.append(get_nonlinearity(self.nonlinearity))
        self.output_shape = shape
            
    def forward(self, input):
        skips = []
        size = input.size()
        out = input
        for m in self.blocks:
            out_prev = out
            out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                skips.append(out_prev)
        return out, skips
    

class decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_conv_blocks, block_type,
                 num_channels_list, output_transform=None, skip=True,
                 dropout=0., normalization=layer_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 kernel_size=3, upsample_mode='conv', init='kaiming_normal_',
                 nonlinearity='ReLU', long_skip_merge_mode=None, ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.output_transform = output_transform
        if not hasattr(output_transform, '__len__'):
            self.output_transform = [output_transform]
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.long_skip_merge_mode = long_skip_merge_mode
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        # Compute all intermediate conv shapes by working backward from the 
        # output shape.
        self._shapes = [self.output_shape,
                        (self.num_channels_list[-1],)+self.output_shape[1:]]
        for i in range(2, self.num_conv_blocks):
            shape_spatial = np.array(self._shapes[-1][1:])//2
            shape = (self.num_channels_list[-i],)+tuple(shape_spatial)
            self._shapes.append(shape)
        self._shapes.append(self.input_shape)
        self._shapes = self._shapes[::-1]
        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = shape[0]
        for n in range(self.num_conv_blocks):
            upsample = bool(n<self.num_conv_blocks-1)    # Not on last layer.
            def _select(a, b=None):
                return a if n>0 else b
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[n],
                                    upsample=upsample,
                                    upsample_mode=self.upsample_mode,
                                    skip=_select(self.skip, False),
                                    dropout=self.dropout,
                                    normalization=_select(self.normalization),
                                    conv_padding=self.conv_padding,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=_select(self.nonlinearity),
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[n]
            if upsample:
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = conv_block(in_channels=self.num_channels_list[n+1],
                                 num_filters=1,
                                 skip=False,
                                 normalization=instance_normalization,
                                 nonlinearity=None,
                                 kernel_size=1,
                                 init=self.init,
                                 ndim=self.ndim)
                    self.cats.append(cat)
                    last_channels += 1
                elif self.long_skip_merge_mode=='cat':
                    last_channels *= 2
                else:
                    pass
            
        '''
        Final output - change number of channels.
        '''
        self.out_nlin = nn.ModuleList()
        self.out_norm = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        for _ in self.output_transform:
            if normalization is not None:
                out_norm = normalization(num_features=last_channels,
                                         **self.norm_kwargs)
            out_nlin = get_nonlinearity(self.nonlinearity)
            out_conv = convolution(in_channels=last_channels,
                                   out_channels=self.output_shape[0],
                                   kernel_size=7,
                                   stride=1,
                                   padding=3,
                                   padding_mode=self.padding_mode,
                                   init=self.init)
            self.out_norm.append(out_norm)
            self.out_nlin.append(out_nlin)
            self.out_conv.append(out_conv)
        
    def forward(self, x, skip_info=None, transform_index=0):
        out = x
        skip_info = skip_info[::-1]
        for n, block in enumerate(self.blocks):
            shape_in  = self._shapes[n]
            shape_out = self._shapes[n+1]
            spatial_shape_in = tuple(max(out.size(i+1),
                                         shape_out[i]-shape_in[i])
                                     for i in range(1, self.ndim+1))
            if np.any(np.less_equal(spatial_shape_in, 0)):
                spatial_shape_in = shape_in[1:]
            out = adjust_to_size(out, spatial_shape_in)
            
            if not out.is_contiguous():
                out = out.contiguous()
            out = block(out)
            out = adjust_to_size(out, shape_out[1:])
            if not out.is_contiguous():
                out = out.contiguous()
            if (self.long_skip_merge_mode is not None and skip_info is not None
                                                      and n<len(skip_info)):
                skip = skip_info[n]
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = self.cats[n]
                    out = torch.cat([out, cat(skip)], dim=1)
                elif self.long_skip_merge_mode=='cat':
                    out = torch.cat([out, skip], dim=1)
                elif self.long_skip_merge_mode=='sum':
                    out = out+skip
                else:
                    raise ValueError("Skip merge mode unrecognized \'{}\'."
                                     "".format(self.long_skip_merge_mode))
        
        out = self.out_norm[transform_index](out)
        out = self.out_conv[transform_index](out)
        out_func = self.output_transform[transform_index]
        if out_func is not None:
            out = out_func(out)
        return out
    
    
class mlp(nn.Module):
    def __init__(self, n_layers, n_input, n_output, n_hidden=None,
                 init='kaiming_normal_', output_transform=None):
        super(mlp, self).__init__()
        assert(n_layers > 0)
        self.n_layers = n_layers
        self.n_input  = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.output_transform = output_transform
        # TODO: support `init` argument.
        if n_hidden is None:
            self.n_hidden = n_output
        layers = []
        if n_layers > 1:
            layers =  [nn.Linear(in_features=n_input, out_features=n_hidden)]
            layers += [nn.Linear(in_features=n_hidden,
                                 out_features=n_hidden)]*max(0, n_layers-2)
            layers += [nn.Linear(in_features=n_hidden, out_features=n_output)]
        else:
            layers = [nn.Linear(in_features=n_input, out_features=n_output)]
        self.model = nn.Sequential(*tuple(layers))
        for layer in self.model:
            if isinstance(layer, nn.Linear) and init is not None:
                layer.weight.data = get_initializer(init)(layer.weight.data)
    def forward(self, x):
        out = self.model(x)
        if self.output_transform is not None:
            out = self.output_transform(out)
        return out


class conv_block(block_abstract):
    """
    A single basic 3x3 convolution.
    Unlike in tiny_block, stride instead of maxpool and upsample before conv.
    """
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(conv_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=2,
                                    init=init)]
        stride = 1
        if subsample:
            stride = 2
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                stride=stride,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlin=nonlinearity)]
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
    
    
def dist_ratio_mse_abs(x, y, eps=1e-7):
    return torch.mean((x-y)**2) / (torch.mean(torch.abs(x-y))+eps)
    

def bce(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    return nn.BCELoss(reduce=reduce)(prediction, target)


def mse(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    return nn.MSELoss(reduce=reduce)(prediction, target)


def mae(prediction, target, reduce=False):
    loss = torch.abs(prediction-target)
    if reduce:
        loss = torch.mean(loss)
    return loss


def grad_norm(module):
    """
    Count the number of parameters in a module.
    """
    parameters = filter(lambda p: p.grad is not None, module.parameters())
    norm = sum([torch.norm(p.grad) for p in parameters])
    return norm


class gan_objective(object):
    def __init__(self, objective, relativistic=False,
                 grad_penalty_real=None, grad_penalty_fake=None,
                 grad_penalty_mean=0):
        # jenson_shannon
        # least_squares
        # hinge
        # wasserstein
        self.objective = objective
        self.relativistic = relativistic
        if relativistic:
            raise NotImplementedError("relativistic=True")
        self.grad_penalty_real = grad_penalty_real
        self.grad_penalty_fake = grad_penalty_fake
        self.grad_penalty_mean = grad_penalty_mean
        if   objective=='jenson_shannon':
            self._D1 = bce(torch.sigmoid(x), 1)
            self._D0 = bce(torch.sigmoid(x), 0)
            self._G  = bce(torch.sigmoid(x), 1)
        elif objective=='least_squares':
            self._D1 = mse(x, 1)
            self._D0 = mse(x, 0)
            self._G  = mse(x, 1)
        elif objective=='hinge':
            self._D1 = lambda x : nn.ReLU()(1.-x)
            self._D0 = lambda x : nn.ReLU()(1.+x)
            self._G  = lambda x : -x
        elif objective=='wasserstein':
            raise NotImplementedError("objective=wasserstein")
        else:
            raise ValueError("Unknown objective: {}".format(objective))
    
    def G(self, disc, fake, real=None):
        return self._foreach(self._G, disc(fake))
    
    def D(self, disc, fake, real):
        loss_real = self._D(real, disc, self._D1, self.grad_penalty_real)
        loss_fake = self._D(fake, disc, self._D0, self.grad_penalty_fake)
        return loss_real+loss_fake
    
    def _D(self, x, disc, objective, grad_penalty_weight):
        if grad_penalty_weight is None:
            return self._foreach(objective, disc(x))
        # Gradient penalty for each item in disc output.
        x.requires_grad = True
        def objective_gp(disc_out):
            grad = torch.autograd.grad(disc_out.sum(),
                                       x,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            norm2 = (grad.view(grad.size()[0],-1)**2).sum(-1)
            norm2 = torch.mean(norm2)
            if self.grad_penalty_mean:
                penalty = (torch.sqrt(norm2)-self.grad_penalty_mean)**2
            else:
                penalty = norm2
            loss = objective(x)+grad_penalty_weight*penalty
            return loss
        return self._foreach(objective_gp, disc(x))
    
    def _foreach(self, func, x):
        # If x is a list, process every element (and reduce to batch dim).
        # (For multi-scale discriminators).
        if not isinstance(x, torch.Tensor):
            return sum([self._foreach(func, elem) for elem in x])
        out = func(x)
        if out.dim()<=1:
            return out
        return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.
