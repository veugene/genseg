import torch
from torch import nn
import numpy as np
from fcn_maker.blocks import (get_nonlinearity,
                              get_initializer,
                              convolution,
                              block_abstract,
                              identity_block,
                              basic_block,
                              tiny_block,
                              dense_block,
                              repeat_block,
                              adjust_to_size)

 
def get_output_shape(layer, input_shape):
    '''
    Works for `convolution`, `nn.Linear`, `identity_block`, `basic_block`,
    `tiny_block`, `dense_block`, `repeat_block`.
    
    `input_shape` is without batch dimension.
    '''
    def compute_conv_out_shape(input_shape, out_channels, padding,
                               kernel_size, stride=1):
        out_shape = 1 + ( np.array(input_shape)[1:]
                         +2*np.array(padding)
                         -np.array(kernel_size)) // np.array(stride)
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_tconv_out_shape(input_shape, out_channels, padding,
                                kernel_size, stride=1):
        out_shape = ( (np.array(input_shape[1:])-1)*np.array(stride)
                     -2*np.array(padding)
                     +np.array(kernel_size))
        out_shape = (out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_pool_out_shape(input_shape, padding,
                               stride=2, ceil_mode=False):
        input_spatial_shape_padded = ( np.array(input_shape)[1:]
                                      +2*np.array(padding))
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
                                           padding=int(layer.conv_padding),
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
                                           padding=layer.op.padding,
                                           kernel_size=layer.op.kernel_size,
                                           stride=layer.op.stride)
        return out_shape
    elif isinstance(layer, nn.Linear):
        return (layer.out_features,)
    elif isinstance(layer, identity_block):
        return input_shape
    elif isinstance(layer, basic_block):
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           out_channels=layer.out_channels,
                                           padding=int(layer.conv_padding),
                                           kernel_size=3,
                                           stride=2 if layer.subsample else 1)
        if layer.upsample:
            out_shape = compute_block_upsample(layer, out_shape)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=int(layer.conv_padding),
                                           kernel_size=3,
                                           stride=1)
        return out_shape
    elif isinstance(layer, tiny_block):
        out_shape = input_shape
        if layer.subsample:
            out_shape = compute_pool_out_shape(input_shape=input_shape,
                                               padding=0,
                                               stride=2)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           out_channels=layer.out_channels,
                                           padding=int(layer.conv_padding),
                                           kernel_size=3,
                                           stride=1)
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


"""
Helper to build a norm -> ReLU -> fully-connected.
"""
class norm_nlin_fc(torch.nn.Module):
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
                 conv_padding=True, vector_out=False, init='kaiming_normal_',
                 nonlinearity='ReLU', ndim=2):
        super(encoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks+int(vector_out==True):
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks "
                             "(+1 if `vector_out==True`).")
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.vector_out = vector_out
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels = input_shape[0]
        self.out_channels = self.num_channels_list[-1]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        self.layers = nn.ModuleList()
        shape = self.input_shape
        last_channels = self.in_channels
        for i in range(self.num_conv_blocks):
            normalization = self.normalization if i else None
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[i],
                                    subsample=bool(i>0),
                                    skip=self.skip,
                                    dropout=self.dropout,
                                    normalization=normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
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
        if self.vector_out:
            self.fc = norm_nlin_fc(in_features=int(np.product(shape)),
                                   out_features=self.num_channels_list[-1],
                                   normalization=None,
                                   norm_kwargs=self.norm_kwargs,
                                   init=self.init,
                                   nonlinearity=self.nonlinearity)
            shape = get_output_shape(self.fc._modules['fc'], shape)
        self.output_shape = shape
            
    def forward(self, input):
        out = input
        for m in self.blocks:
            out = m(out)
        if self.vector_out:
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        out = nn.functional.relu(out)
        return out
    

class decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_conv_blocks, block_type,
                 num_channels_list, output_transform=None, skip=True,
                 dropout=0., normalization=instance_normalization,
                 norm_kwargs=None, conv_padding=True, vector_in=False,
                 upsample_mode='conv', init='kaiming_normal_',
                 nonlinearity='ReLU', ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks+int(vector_in==True):
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks "
                             "(+1 if `vector_in==True`).")
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.output_transform = output_transform
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.vector_in = vector_in
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        # Compute all intermediate conv shapes by working backward from the 
        # output shape.
        self._shapes = [self.output_shape,
                        (self.num_channels_list[-2],)+self.output_shape[1:]]
        for i in range(2, self.num_conv_blocks):
            s = np.array(self._shapes[-1][1:])
            shape = (self.num_channels_list[-i-1],)+tuple((s+s%2)//2)
            self._shapes.append(shape)
        self._shapes.append(self.input_shape)
        self._shapes = self._shapes[::-1]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        if self.vector_in:
            out_features = ( self.num_channels_list[0]
                            *int(np.product(self._shapes[0][1:])))
            self.fc = norm_nlin_fc(in_features=self.in_channels,
                                   out_features=out_features,
                                   normalization=None,
                                   norm_kwargs=self.norm_kwargs,
                                   init=self.init,
                                   nonlinearity=self.nonlinearity)
            last_channels = out_features
            shape = self._shapes[0]
        last_channels = shape[0]
        for i in range(self.num_conv_blocks, 0, -1):
            upsample = bool(i>1)    # Not on last layer.
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[-i],
                                    upsample=upsample,
                                    upsample_mode=self.upsample_mode,
                                    skip=self.skip,
                                    dropout=self.dropout,
                                    normalization=self.normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[-i]
            
        '''
        Final output - change number of channels.
        '''
        self.output = convolution(in_channels=last_channels,
                                  out_channels=self.output_shape[0],
                                  kernel_size=1,
                                  ndim=self.ndim)
        self.output_shape = shape
        
    def forward(self, x):
        out = x
        if self.vector_in:
            out = self.fc(out)
            out = out.view(out.size(0), *self._shapes[0])
        for m, shape_in, shape_out in zip(self.blocks,
                                          self._shapes[:-1],
                                          self._shapes[1:]):
            spatial_shape_in = tuple(max(out.size(i+1),
                                         shape_out[i]-shape_in[i])
                                     for i in range(1, self.ndim+1))
            if np.any(np.less_equal(spatial_shape_in, 0)):
                spatial_shape_in = shape_in[1:]
            out = adjust_to_size(out, spatial_shape_in)
            if not out.is_contiguous():
                out = out.contiguous()
            out = m(out)
            out = adjust_to_size(out, shape_out[1:])
            if not out.is_contiguous():
                out = out.contiguous()
        out = self.output(out)
        if self.output_transform is not None:
            out = self.output_transform(out)
        return out
    
    
class mlp(nn.Module):
    def __init__(self, n_input, n_output, n_layers, n_hidden=None,
                 init='kaiming_normal_'):
        super(mlp, self).__init__()
        assert(n_layers > 0)
        self.n_input  = n_input
        self.n_output = n_output
        self.n_layers = n_layers
        self.n_hidden = n_hidden
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
        return self.model(x)
    
    
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
    
    
if __name__=='__main__':
    image_shape = (3, 256, 171, 91)
    vector_len = 50
    batch_size = 2
    
    # TEST: encoder
    model = encoder(input_shape=image_shape,
                    num_conv_blocks=6,
                    block_type=basic_block,
                    num_channels_list=[10, 20, 30, 40, 50, 60, vector_len],
                    skip=True,
                    dropout=0.,
                    normalization=instance_normalization,
                    norm_kwargs=None,
                    conv_padding=True,
                    vector_out=True,
                    init='kaiming_normal_',
                    nonlinearity='ReLU',
                    ndim=len(image_shape)-1)
    model = model.cuda()
    test_input = np.random.rand(batch_size, *image_shape).astype(np.float32)
    test_input = torch.autograd.Variable(torch.from_numpy(test_input)).cuda()
    output = model(test_input)
    print("ENCODER: {} parameters"
          "".format(sum([np.prod(p.size()) for p in model.parameters()])))
    print("encoder: input_shape={}, output_shape={}"
          "".format((batch_size,)+image_shape, tuple(output.size())))
    
    # TEST: decoder
    model = decoder(input_shape=(vector_len,),
                    output_shape=image_shape,
                    num_conv_blocks=6,
                    block_type=basic_block,
                    num_channels_list=[60, 50, 40, 30, 20, 10, 3],
                    skip=True,
                    dropout=0.,
                    normalization=instance_normalization,
                    norm_kwargs=None,
                    conv_padding=True,
                    vector_in=True,
                    init='kaiming_normal_',
                    nonlinearity='ReLU',
                    ndim=len(image_shape)-1)
    model = model.cuda()
    test_input = np.random.rand(batch_size, vector_len).astype(np.float32)
    test_input = torch.autograd.Variable(torch.from_numpy(test_input)).cuda()
    output = model(test_input)
    print("DECODER: {} parameters"
          "".format(sum([np.prod(p.size()) for p in model.parameters()])))
    print("decoder: input_shape={}, output_shape={}"
          "".format((batch_size, vector_len), tuple(output.size())))
    
