#  Semi Supervised Segmentation using Mean Teacher Approach
#  Implements: https://arxiv.org/abs/1807.04657

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
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
                                        norm_nlin_conv,
                                        pool_block,
                                        repeat_block)
from model.mean_teacher_segmentation import segmentation_model


def build_model(long_skip='skinny_cat', lambda_con=0.01, alpha_max=0.99):
    if long_skip=="none":
        long_skip = None
    N = 512 # Number of features at the bottleneck.
    kwargs = {
        'in_channels'         : 4,
        'block_type'          : conv_block,
        'enc_layer_size'      : [N//32, N//16, N//8, N//4, N//2, N],
        'dec_layer_size'      : [N//2, N//4, N//8, N//16, N//32],
        'skip'                : True,
        'long_skip_merge_mode': long_skip,
        'dropout'             : 0.,
        'enc_normalization'   : instance_normalization,
        'dec_normalization'   : layer_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'skip_pool_indices'   : False,
        'ndim'                : 2}
    model = segmentation_model(
        student=encoder_decoder(**kwargs),
        teacher=encoder_decoder(**kwargs),
        loss_seg=dice_loss([1,2,4]),
        lambda_con=lambda_con,
        alpha_max=alpha_max)
    return {'G': model}


class encoder_decoder(nn.Module):
    def __init__(self, in_channels, block_type,
                 enc_layer_size, dec_layer_size, skip=True,
                 long_skip_merge_mode=None, dropout=0.,
                 enc_normalization=instance_normalization,
                 dec_normalization=layer_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', upsample_mode='repeat',
                 nonlinearity='ReLU', skip_pool_indices=False, ndim=2):
        super(encoder_decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        self.in_channels = in_channels
        self.block_type = block_type
        self.enc_layer_size = enc_layer_size
        self.dec_layer_size = dec_layer_size
        self.skip = skip
        self.long_skip_merge_mode = long_skip_merge_mode
        self.dropout = dropout
        self.enc_normalization = enc_normalization
        self.dec_normalization = dec_normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.upsample_mode = upsample_mode
        self.nonlinearity = nonlinearity
        self.skip_pool_indices = skip_pool_indices
        self.ndim = ndim
        
        '''
        Set up ENCODER blocks.
        '''
        self.enc_blocks = nn.ModuleList()
        last_channels = self.in_channels
        conv = convolution(in_channels=last_channels,
                           out_channels=self.enc_layer_size[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode=self.padding_mode,
                           init=self.init)
        self.enc_blocks.append(conv)
        last_channels = self.enc_layer_size[0]
        for i in range(1, len(self.enc_layer_size)):
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.enc_layer_size[i],
                                    subsample=True,
                                    skip=skip,
                                    dropout=self.dropout,
                                    normalization=self.enc_normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.enc_blocks.append(block)
            last_channels = self.enc_layer_size[i]
        if enc_normalization is not None:
            block = enc_normalization(ndim=self.ndim,
                                  num_features=last_channels,
                                  **self.norm_kwargs)
            self.enc_blocks.append(block)
        self.enc_blocks.append(get_nonlinearity(self.nonlinearity))
        
        '''
        Set up DECODER blocks.
        '''
        self.cats   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for n in range(len(self.dec_layer_size)):
            def _select(a, b=None):
                return a if n>0 else b
            block = self.block_type(
                in_channels=last_channels,
                num_filters=self.dec_layer_size[n],
                upsample=True,
                upsample_mode=self.upsample_mode,
                skip=_select(self.skip, False),
                dropout=self.dropout,
                normalization=_select(self.dec_normalization),
                norm_kwargs=self.norm_kwargs,
                padding_mode=self.padding_mode,
                kernel_size=self.kernel_size,
                init=self.init,
                nonlinearity=_select(self.nonlinearity),
                ndim=self.ndim)
            self.dec_blocks.append(block)
            last_channels = self.dec_layer_size[n]
            if   self.long_skip_merge_mode=='skinny_cat':
                cat = conv_block(in_channels=self.dec_layer_size[n],
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
        self.pre_conv = norm_nlin_conv(in_channels=last_channels,
                                       out_channels=last_channels,  # NOTE
                                       kernel_size=self.kernel_size,
                                       init=self.init,              # NOTE
                                       normalization=self.dec_normalization,
                                       norm_kwargs=self.norm_kwargs,
                                       nonlinearity=self.nonlinearity,
                                       padding_mode=self.padding_mode)
        kwargs_out_conv = {
            'in_channels': last_channels,
            'out_channels': 1,      # TODO : fix this in all implementations
            'kernel_size': 7,
            'normalization': self.dec_normalization,
            'norm_kwargs': self.norm_kwargs,
            'nonlinearity': self.nonlinearity,
            'padding_mode': self.padding_mode,
            'init': self.init,
            'ndim': self.ndim}
        self.out_conv = norm_nlin_conv(**kwargs_out_conv)
        
        # Classifier for segmentation.
        self.classifier = convolution(
            in_channels=1,          # TODO : this is dumb, fix this
            out_channels=1,
            kernel_size=1,
            ndim=self.ndim)        
            
    def forward(self, input):
        
        # ENCODE
        skips = []
        size = input.size()
        out = input
        for m in self.enc_blocks:
            out_prev = out
            if isinstance(m, pool_block):
                out, indices = m(out_prev)
            else:
                out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                if self.skip_pool_indices:
                    skips.append(indices)
                else:
                    skips.append(out_prev)
        
        # DECODE
        if skips is not None:
            skips = skips[::-1]
        for n, block in enumerate(self.dec_blocks):
            if (self.long_skip_merge_mode=='pool' and skips is not None
                                                  and n<len(skips)):
                skip = skips[n]
                out = adjust_to_size(out, skip.size()[2:])
                out = block(out, unpool_indices=skip)
            elif (self.long_skip_merge_mode is not None
                                                  and skips is not None
                                                  and n<len(skips)):
                skip = skips[n]
                out = block(out)
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
            else:
                out = block(out)
                out = adjust_to_size(out, skips[n].size()[2:])
            if not out.is_contiguous():
                out = out.contiguous()
        out = self.pre_conv(out)
        out = self.out_conv(out)
        
        # CLASSIFY
        out = self.classifier(out)
        out = torch.sigmoid(out)
        return out
