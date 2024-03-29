import torch
from torch import nn
from torch.functional import F
import numpy as np
from fcn_maker.blocks import (adjust_to_size,
                              basic_block,
                              tiny_block,
                              convolution,
                              norm_nlin_conv,
                              get_initializer,
                              get_nonlinearity)
from fcn_maker.loss import dice_loss
from model.common.network.basic import (get_output_shape,
                                        conv_block,
                                        batch_normalization,
                                        instance_normalization,
                                        layer_normalization)
from model.common.losses import dist_ratio_mse_abs
from model.ae_segmentation import segmentation_model


def build_model():
    N = 1024 # Number of features at the bottleneck.
    image_size = (1, 48, 48)
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 5,
        'block_type'          : conv_block,
        'num_channels_list'   : [N//16, N//8, N//4, N//2, N],
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 4,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    encoder_instance = encoder(**encoder_kwargs)
    enc_out_shape = encoder_instance.output_shape
    
    decoder_kwargs = {
        'input_shape'         : enc_out_shape,
        'output_shape'        : image_size,
        'num_conv_blocks'     : 5,
        'block_type'          : conv_block,
        'num_channels_list'   : [N, N//2, N//4, N//8, N//16],
        'output_transform'    : torch.sigmoid,
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 5,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    
    print("DEBUG: sample_shape={}".format(enc_out_shape))

    model = segmentation_model(encoder=encoder_instance,
                               decoder_rec=decoder(
                                   long_skip_merge_mode=None,
                                   **decoder_kwargs),
                               decoder_seg=decoder(
                                   long_skip_merge_mode='skinny_cat',
                                   **decoder_kwargs),
                               loss_rec=dist_ratio_mse_abs,
                               loss_seg=dice_loss(),
                               lambda_rec=1.,
                               lambda_seg=10.,
                               rng=np.random.RandomState(1234))
    
    return {'G': model}


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
        
    def forward(self, z, skip_info=None, out_idx=0):
        out = z
        if skip_info is not None:
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
        out = self.out_norm[out_idx](out)
        out = self.out_conv[out_idx](out)
        out_func = self.output_transform[out_idx]
        if out_func is not None:
            out = out_func(out)
        return out
