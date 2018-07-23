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
                              repeat_block)

 
def get_output_shape(layer, input_shape):
    '''
    Works for `convolution`, `nn.Linear`, `identity_block`, `basic_block`,
    `tiny_block`, `dense_block`, `repeat_block`.
    
    `input_shape` is without batch dimension.
    '''
    def compute_conv_out_shape(input_shape, padding, kernel_size, stride=1):
        out_shape = 1 + ( np.array(input_shape)[1:]
                         +2*np.array(padding)
                         -np.array(kernel_size)) / np.array(stride)
        out_shape = (layer.out_channels,)+tuple(out_shape)
        return out_shape
    
    def compute_pool_out_shape(input_shape, padding, stride=2):
        input_spatial_shape_padded = ( np.array(input_shape)[1:]
                                      +2*np.array(padding))
        out_shape = ( input_spatial_shape_padded//stride,
                     +input_spatial_shape_padded%stride)
        out_shape = (input_shape[0],)+tuple(out_shape)
        return out_shape
    
    if isinstance(layer, block_abstract):
        if layer.upsample:
            raise NotImplementedError("Shape inference not implemented for "
                                      "upsampling blocks.")
    if isinstance(layer, convolution):
        out_shape = compute_conv_out_shape(input_shape=input_shape,
                                           padding=layer.padding,
                                           kernel_size=layer.kernel_size,
                                           stride=layer.stride)
        return out_shape
    elif isinstance(layer, nn.Linear):
        return (layer.out_channels,)+tuple(input_shape[1:])
    elif isinstance(layer, identity_block):
        return input_shape
    elif isinstance(layer, basic_block):
        padding = layer.kernel_size//2 if layer.conv_padding else 0
        shape1 = compute_conv_out_shape(input_shape=input_shape,
                                        padding=padding,
                                        kernel_size=3,
                                        stride=2 if layer.subsample else 1)
        shape2 = compute_conv_out_shape(input_shape=shape1,
                                        padding=padding,
                                        kernel_size=3,
                                        stride=1)
        return shape2
    elif isinstance(layer, tiny_block):
        padding = layer.kernel_size//2 if layer.conv_padding else 0
        out_shape = input_shape
        if layer.subsample:
            out_shape = compute_pool_out_shape(input_shape=input_shape,
                                               padding=0,
                                               stride=2)
        out_shape = compute_conv_out_shape(input_shape=out_shape,
                                           padding=padding,
                                           kernel_size=3,
                                           stride=2 if layer.subsample else 1)
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
    def __init__(self, in_channels, out_channels, nonlinearity='ReLU',
                 normalization=instance_normalization, norm_kwargs=None,
                 init='kaiming_normal'):
        super(norm_nlin_conv, self).__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.init = init
        if normalization is not None:
            self._modules['norm'] = normalization(ndim=ndim,
                                                  num_features=in_channels,
                                                  **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        self._modules['fc'] = nn.Linear(in_channels=in_channels,
                                        out_channels=out_channels)

    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out


class image_to_vector(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, num_fc_layers,
                 block_type, num_channels_list, short_skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 conv_padding=True, init='kaiming_normal',
                 nonlinearity='ReLU', ndim=2):
        super(image_to_vector, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels)!=num_conv_blocks+num_fc_layers:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks and fc layers")
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.num_fc_layers = num_fc_layers
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.short_skip = short_skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
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
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[i],
                                    subsample=bool(i>0),
                                    skip=self.short_skip,
                                    dropout=self.dropout,
                                    normalization=self.normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[i]
        for i in range(self.num_conv_blocks,
                       self.num_conv_blocks+self.num_fc_layers):
            layer = norm_nlin_fc(in_channels=np.product(shape),
                                 out_channels=self.num_channels_list[i],
                                 normalization=self.normalization,
                                 norm_kwargs=self.norm_kwargs,
                                 init=self.init,
                                 nonlinearity=self.nonlinearity)
            self.layers.append(layer)
            shape = get_output_shape(layer._modules['fc'], shape)
        
    def forward(self, input):
        out = input
        for m in self.blocks:
            out = m(out)
        out = out.view(out.size(0), -1)
        for m in self.layers:
            out = m(out)
        return out
    

class vector_to_image(nn.Module):
    pass

