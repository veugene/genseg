# 003 with new contrastive loss
# 004 using the base model

import numpy as np
import torch

from fcn_maker.loss import dice_loss
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (basic_block,
                              batch_normalization,
                              bottleneck,
                              convolution,
                              instance_normalization,
                              norm_nlin_conv,
                              tiny_block)
from model.ae_segmentation import segmentation_model


def build_model(lambda_rec=0, lambda_cl=1, temperature=0.1, input_size=256):
    kwargs = {
        'in_channels':           2,
        'num_classes':           1,
        'num_init_blocks':       3,
        'num_main_blocks':       3,
        'main_block_depth':      1,
        'init_num_filters':      64,
        'short_skip':            True,
        'long_skip':             True,
        'long_skip_merge_mode': 'sum',
        'main_block':           basic_block,
        'init_block':           basic_block,
        'upsample_mode':        'conv',
        'dropout':              0,
        'normalization':        batch_normalization,
        'norm_kwargs':          None,
        'conv_padding':         True,
        'padding_mode':         'constant',
        'kernel_size':          3,
        'init':                 'kaiming_normal_',
        'nonlinearity':         'ReLU',
        'ndim':                 2}
    
    # Compute bottlneck size.
    n_downscale = kwargs['num_init_blocks']+kwargs['num_main_blocks']
    xy = 256//2**n_downscale
    n_filters = kwargs['init_num_filters']*2**(kwargs['num_main_blocks']-1)
    if kwargs['main_block']==bottleneck:
        n_filters *= 4
    bottleneck_size = (n_filters, xy, xy)
        
    # Prepare segmentation model.
    fcn = assemble_resunet(**kwargs)
    encoder = fcn.encoder
    decoder_seg = fcn.decoder
    
    # ResUNet.
    model = segmentation_model(encoder=encoder,
                               decoder_seg=decoder_seg,
                               decoder_rec=None,
                               loss_seg=dice_loss(),
                               lambda_rec=0.,
                               lambda_seg=1.,
                               rng=np.random.RandomState(1234))
    return {'G': model}
