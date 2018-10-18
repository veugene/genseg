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
                          batch_normalization,
                          instance_normalization)
from model.gan_debug import gan
from model.mine import mine


def build_model():
    image_size = (1, 30, 30)
    bottleneck_size = (100, 1, 1)
    N = 1024
    
    class discriminator(nn.Module):
        def __init__(self, *args, **kwargs):
            super(discriminator, self).__init__()
            self.encoder = encoder(*args, **kwargs)
            self.nlin = get_nonlinearity(self.encoder.nonlinearity)
            self.final_conv = convolution(
                in_channels=self.encoder.out_channels,
                out_channels=1,
                kernel_size=1,
                init=self.encoder.init)
            
        def forward(self, x):
            out = self.encoder(x)
            out = self.nlin(out)
            out = self.final_conv(out)
            out = torch.sigmoid(out)
            return out
        
    class g_decoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super(g_decoder, self).__init__()
            self.tconv = nn.ConvTranspose2d(bottleneck_size[0]*3, N,
                                            kernel_size=4,
                                            stride=2,
                                            padding=0)
            self.model = decoder(*args, **kwargs)
        def forward(self, common, residual, unique):
            x = torch.cat([common, residual, unique], dim=1)
            x = self.tconv(x)
            x = self.model(x)
            return x
    
    decoder_kwargs = {
        'input_shape'       : (N,4,4),
        'output_shape'      : image_size,
        'num_conv_blocks'   : 4,
        'block_type'        : tiny_block,
        'num_channels_list' : [N//2, N//4, N//8, 1],
        'output_transform'  : torch.sigmoid,
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_in'         : False,
        'init'              : 'kaiming_normal_',
        'upsample_mode'     : 'repeat',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    discriminator_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : tiny_block,
        'num_channels_list' : [N//8, N//8, N//4, N//2, N],
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}

    
    submodel = {
        'g_common'          : lambda x:x,
        'g_residual'        : lambda x:x,
        'g_unique'          : lambda x:x,
        'g_output'          : g_decoder(**decoder_kwargs),
        'disc'              : discriminator(**discriminator_kwargs)}
    model = gan(**submodel,
                z_size=bottleneck_size,
                z_constant=0)
    
    return model
