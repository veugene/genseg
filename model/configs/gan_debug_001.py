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
                          instance_normalization)
from model.gan_debug import gan
from model.mine import mine


def build_model():
    image_size = (1, 30, 30)
    bottleneck_size = (50, 2, 2)
    
    decoder_kwargs = {
        'input_shape'       : bottleneck_size,
        'output_shape'      : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [50, 50, 30, 20, 1],
        'output_transform'  : torch.sigmoid,
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : instance_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_in'         : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    conv_stack_kwargs = {
        'in_channels'       : 50,
        'out_channels'      : 50,
        'num_blocks'        : 4,
        'block_type'        : basic_block,
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : instance_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    discriminator_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 4,
        'block_type'        : tiny_block,
        'num_channels_list' : [20, 20, 40, 80],
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : None,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    class conv_stack(nn.Module):
        def __init__(self, in_channels, out_channels, num_blocks, block_type,
                     skip=True, dropout=0.,
                     normalization=instance_normalization, norm_kwargs=None,
                     conv_padding=True, init='kaiming_normal_', 
                     nonlinearity='ReLU', ndim=2):
            super(conv_stack, self).__init__()
            self.in_channels   = in_channels
            self.out_channels  = out_channels
            self.num_blocks    = num_blocks
            self.block_type    = block_type
            self.skip          = skip
            self.dropout       = dropout
            self.normalization = normalization
            self.norm_kwargs   = norm_kwargs if norm_kwargs is not None else {}
            self.conv_padding  = conv_padding
            self.nonlinearity  = nonlinearity
            self.ndim          = ndim
            self.init          = init
            assert num_blocks > 0
            block_kwargs = {'num_filters': self.out_channels,
                            'skip': self.skip,
                            'dropout': self.dropout,
                            'normalization': self.normalization,
                            'norm_kwargs': self.norm_kwargs,
                            'conv_padding': self.conv_padding,
                            'init': self.init,
                            'nonlinearity': self.nonlinearity,
                            'ndim': self.ndim}
            blocks = [block_type(in_channels=self.in_channels,
                                 **block_kwargs)]
            for i in range(num_blocks-1):
                blocks.append(block_type(in_channels=self.out_channels,
                                         **block_kwargs))
            self.blocks = nn.Sequential(*tuple(blocks))
        def forward(self, x):
            return self.blocks(x)
        
    class patch_discriminator(nn.Module):
        def __init__(self, *args, **kwargs):
            super(patch_discriminator, self).__init__()
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
            self.model = decoder(*args, **kwargs)
        def forward(self, common, residual, unique):
            x = sum([common, residual, unique])
            return self.model(x)
    
    submodel = {
        'g_common'          : conv_stack(**conv_stack_kwargs),
        'g_residual'        : conv_stack(**conv_stack_kwargs),
        'g_unique'          : conv_stack(**conv_stack_kwargs),
        'g_output'          : g_decoder(**decoder_kwargs),
        'disc'              : patch_discriminator(**discriminator_kwargs)}
    model = gan(**submodel,
                z_size=bottleneck_size,
                z_constant=0,
                grad_penalty=1.,
                disc_clip_norm=None)
    
    return model
