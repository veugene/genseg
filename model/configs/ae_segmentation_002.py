import torch
from torch import nn
import numpy as np
from fcn_maker.blocks import (basic_block,
                              tiny_block,
                              convolution,
                              get_initializer,
                              get_nonlinearity)
from fcn_maker.loss import dice_loss
from model.common import (encoder,
                          decoder,
                          instance_normalization,
                          mae)
from model.ae_segmentation import segmentation_model


def build_model():
    image_size = (1, 100, 100)
    bottleneck_size = (50, 7, 7)

    encoder_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [20, 20, 30, 50, 50],
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : instance_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    decoder_kwargs = {
        'input_shape'       : bottleneck_size,
        'output_shape'      : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [50, 50, 30, 20, 1],
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : instance_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_in'         : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    model = segmentation_model(encoder=encoder(**encoder_kwargs),
                               decoder_rec=decoder(**decoder_kwargs),
                               decoder_seg=decoder(**decoder_kwargs),
                               loss_rec=mae,
                               loss_seg=dice_loss(),
                               lambda_rec=0.,
                               lambda_seg=10.,
                               rng=np.random.RandomState(1234))
    
    return model
