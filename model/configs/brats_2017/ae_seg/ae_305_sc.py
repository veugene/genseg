# bds3_300_sc_residual_skinny_cat_repeat but 3D (nnunet-like variant B)
#
# NOTE: much fewer parameters
# NOTE: discriminator has more parameters (like basic model)

from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss

from model.ae_segmentation import segmentation_model
from model.common.losses import mae
from model.common.network.basic import (adjust_to_size,
                                        batch_normalization,
                                        basic_block,
                                        convolution,
                                        conv_block,
                                        do_upsample,
                                        get_initializer,
                                        get_nonlinearity,
                                        get_output_shape,
                                        instance_normalization,
                                        layer_normalization,
                                        munit_discriminator,
                                        norm_nlin_conv,
                                        pool_block,
                                        recursive_spectral_norm,
                                        repeat_block)


def build_model(lambda_rec=1, lambda_seg=1):
    N = 480 # Number of features at the bottleneck.
    n = 128 # Number of features to sample at the bottleneck.
    image_size = (4, 80, 120, 60)
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [24, 48, 96, 192, 384, 480],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'constant',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 3}
    encoder_instance = encoder(**encoder_kwargs)
    enc_out_shape = encoder_instance.output_shape
    
    decoder_common_kwargs = {
        'input_shape'         : (N-n,)+enc_out_shape[1:],
        'out_channels'        : image_size[0],
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [480, 384, 192, 96, 48, 24],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'constant',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'skinny_cat',
        'ndim'                : 3}
    
    decoder_residual_kwargs = {
        'input_shape'         : enc_out_shape,
        'out_channels'        : image_size[0],
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [480, 384, 192, 96, 48, 24],
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'constant',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'skinny_cat',
        'ndim'                : 3}
    
    model = segmentation_model(encoder=encoder_instance,
                               decoder_rec=decoder(
                                   num_classes=None,
                                   **decoder_common_kwargs),
                               decoder_seg=decoder(
                                   num_classes=1,
                                   **decoder_residual_kwargs),
                               loss_rec=mae,
                               loss_seg=dice_loss([1, 2, 4]),
                               lambda_rec=lambda_rec,
                               lambda_seg=lambda_seg,
                               rng=np.random.RandomState(1234))

    return {'G': model}


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
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
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.in_channels = input_shape[0]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        conv = convolution(in_channels=self.in_channels,
                           out_channels=self.num_channels_list[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode=self.padding_mode,
                           init=self.init,
                           ndim=self.ndim)
        self.blocks.append(conv)
        self.out_channels = self.num_channels_list[0]
        self.output_shape = get_output_shape(conv, input_shape)
        self.add_block(num_filters=self.num_channels_list[0], skip=skip)
        for i in range(1, self.num_conv_blocks-1):
            self.add_block(num_filters=self.num_channels_list[i],
                           subsample=True,
                           skip=skip)
            self.add_block(num_filters=self.num_channels_list[i],
                           skip=skip)
        # Bottleneck.
        self.add_block(num_filters=self.num_channels_list[-1],
                       subsample=True,
                       skip=skip)
        if normalization is not None:
            block = normalization(ndim=self.ndim,
                                  num_features=self.out_channels,
                                  **self.norm_kwargs)
            self.blocks.append(block)
        self.blocks.append(get_nonlinearity(self.nonlinearity))
        
    def add_block(self, num_filters, **kwargs):
        block_kwargs = {
            'in_channels': self.out_channels,
            'num_filters': num_filters,
            'dropout': self.dropout,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'padding_mode': self.padding_mode,
            'kernel_size': self.kernel_size,
            'init': self.init,
            'nonlinearity': self.nonlinearity,
            'ndim': self.ndim
        }
        block_kwargs.update(kwargs)
        block = self.block_type(**block_kwargs)
        self.blocks.append(block)
        self.output_shape = get_output_shape(block, self.output_shape)
        self.out_channels = num_filters
            
    def forward(self, input):
        skips = []
        size = input.size()
        out = input
        for m in self.blocks:
            out_prev = out
            if isinstance(m, pool_block):
                out, indices = m(out_prev)
            else:
                out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                skips.append(out_prev)
        return out, skips


class switching_normalization(nn.Module):
    def __init__(self, normalization, *args, **kwargs):
        super(switching_normalization, self).__init__()
        self.norm0 = normalization(*args, **kwargs)
        self.norm1 = normalization(*args, **kwargs)
        self.mode = 0
    def set_mode(self, mode):
        assert mode in [0, 1]
        self.mode = mode
    def forward(self, x):
        if self.mode==0:
            return self.norm0(x)
        else:
            return self.norm1(x)
    

class decoder(nn.Module):
    def __init__(self, input_shape, out_channels, num_conv_blocks, block_type,
                 num_channels_list, num_classes=None, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3, upsample_mode='conv',
                 init='kaiming_normal_', nonlinearity='ReLU',
                 long_skip_merge_mode=None, ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        # long_skip_merge_mode settings.
        valid_modes = [None, 'sum', 'skinny_cat', 'cat']
        if long_skip_merge_mode not in valid_modes:
            raise ValueError("`long_skip_merge_mode` must be one of {}."
                             "".format(", ".join(["\'{}\'".format(mode)
                                                  for mode in valid_modes])))
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.num_classes = num_classes
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.long_skip_merge_mode = long_skip_merge_mode
        self.ndim = ndim
        self.in_channels  = self.input_shape[0]

        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.output_shape = self.input_shape
        self.out_channels = self.input_shape[0]
        self.add_block(num_filters=self.num_channels_list[0],
                       skip=skip,
                       normalization=None,
                       nonlinearity=None)
        for n in range(1, self.num_conv_blocks):
            self.add_block(
                num_filters=self.num_channels_list[n],
                upsample=True,
                upsample_mode=self.upsample_mode,
                skip=self.skip,
                normalization=normalization,
            )
            self.add_block(
                num_filters=self.num_channels_list[n],
                normalization=normalization,
            )
            if   self.long_skip_merge_mode=='skinny_cat':
                cat = conv_block(in_channels=self.num_channels_list[n],
                                 num_filters=1,
                                 skip=False,
                                 normalization=instance_normalization,
                                 nonlinearity=None,
                                 kernel_size=1,
                                 init=self.init,
                                 ndim=self.ndim)
                self.cats.append(cat)
                self.out_channels += 1
            elif self.long_skip_merge_mode=='cat':
                self.out_channels *= 2
            else:
                pass
            
        '''
        Final output - change number of channels.
        '''
        self.pre_conv = norm_nlin_conv(in_channels=self.out_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       init=self.init,
                                       normalization=normalization,
                                       norm_kwargs=self.norm_kwargs,
                                       nonlinearity=self.nonlinearity,
                                       padding_mode=self.padding_mode,
                                       ndim=self.ndim)
        kwargs_out_conv = {
            'in_channels': self.out_channels,
            'out_channels': out_channels,
            'kernel_size': 7,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'nonlinearity': self.nonlinearity,
            'padding_mode': self.padding_mode,
            'init': self.init,
            'ndim': self.ndim}
        self.out_conv = norm_nlin_conv(**kwargs_out_conv)
        self.out_channels = out_channels
        
        # Classifier for segmentation (mode 1).
        if self.num_classes is not None:
            self.classifier = convolution(
                in_channels=self.out_channels,
                out_channels=self.num_classes,
                kernel_size=1,
                ndim=self.ndim)
    
    def add_block(self, num_filters, **kwargs):
        block_kwargs = {
            'in_channels': self.out_channels,
            'num_filters': num_filters,
            'dropout': self.dropout,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'padding_mode': self.padding_mode,
            'kernel_size': self.kernel_size,
            'init': self.init,
            'nonlinearity': self.nonlinearity,
            'ndim': self.ndim
        }
        block_kwargs.update(kwargs)
        block = self.block_type(**block_kwargs)
        self.blocks.append(block)
        self.output_shape = get_output_shape(block, self.output_shape)
        self.out_channels = num_filters
        
    def forward(self, z, skip_info=None):
        # Compute output.
        if skip_info is not None:
            skip_info = skip_info[::-1]
        out = self.blocks[0](z)
        for n in range(self.num_conv_blocks-1):
            s = out.shape
            out = self.blocks[2*n+1](out)
            out = self.blocks[2*n+2](out)
            skip = skip_info[n]
            out = adjust_to_size(out, skip.size()[2:])
            if   self.long_skip_merge_mode=='skinny_cat':
                cat = self.cats[n]
                out = torch.cat([out, cat(skip)], dim=1)
            elif self.long_skip_merge_mode=='cat':
                out = torch.cat([out, skip], dim=1)
            elif self.long_skip_merge_mode=='sum':
                out = out+skip
            else:
                ValueError()
            if not out.is_contiguous():
                out = out.contiguous()
        out = self.pre_conv(out)
        out = self.out_conv(out)
        if self.num_classes is None:
            out = torch.tanh(out)
        else:
            out = self.classifier(out)
            if self.num_classes == 1:
                out = torch.sigmoid(out)
            else:
                out = torch.softmax(out, dim=1)

        return out
