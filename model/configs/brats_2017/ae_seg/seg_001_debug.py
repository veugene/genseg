import torch
from torch import nn
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
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
                                        tiny_block)
from model.common.losses import dist_ratio_mse_abs
from model.ae_segmentation import segmentation_model


def build_model():
    N = 256 # Number of features at the bottleneck.
    image_size = (4, 240, 120)
    
    fcn_kwargs = {
        'in_channels'         : 4,
        'num_classes'         : 1,
        'num_init_blocks'     : 2,
        'num_main_blocks'     : 3,
        'main_block_depth'    : 1,
        'init_num_filters'    : 4,
        'dropout'             : 0.,
        'main_block'          : basic_block,
        'init_block'          : tiny_block,
        'norm_kwargs'         : {'momentum': 0.1},
        'nonlinearity'        : lambda : nn.LeakyReLU(inplace=True),
        'ndim'                : 2}
    
    model = segmentation_model(encoder=encoder(**fcn_kwargs),
                               decoder_rec=None,
                               decoder_seg=segmenter(),
                               #loss_rec=dist_ratio_mse_abs,
                               loss_seg=dice_loss([1,2,4]),
                               lambda_rec=0.,
                               lambda_seg=1.,
                               rng=np.random.RandomState(1234))
    
    return {'G' : model}


class encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(encoder, self).__init__()
        self.fcn = assemble_resunet(*args, **kwargs)
    def forward(self, x):
        out = skip_info = self.fcn(x)
        return out, skip_info


class segmenter(nn.Module):
    def __init__(self):
        super(segmenter, self).__init__()
    def forward(self, x, skip_info):
        return skip_info