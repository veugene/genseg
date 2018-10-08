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
from model.bidomain_segmentation import segmentation_model
from model.mine import mine


def build_model():
    N = 64
    image_size = (1, 30, 30)
    lambdas = {
        'lambda_disc'       : 1,
        'lambda_x_id'       : 1,
        'lambda_z_id'       : 0,
        'lambda_const'      : 0,
        'lambda_cyc'        : 1,
        'lambda_mi'         : 0,
        'lambda_seg'        : 0}
    
    class f_factor(nn.Module):
        def __init__(self, *args, **kwargs):
            super(f_factor, self).__init__()
            self.model = encoder(*args, **kwargs)
            # Compute output shape (two shapes, split along channel dim)
            _s = self.model.output_shape
            self.output_shape = ( (_s[0]//2,)+_s[1:],
                                  (_s[0]-_s[0]//2,)+_s[1:] )
        def forward(self, x):
            a, b = torch.chunk(self.model(x), 2, dim=1)
            if not a.is_contiguous():
                a = a.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
            return a, b
    
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
    
    
    class mi_estimation_network(nn.Module):
        def __init__(self, x_size, z_size, n_hidden):
            super(mi_estimation_network, self).__init__()
            self.x_size = x_size
            self.z_size = z_size
            self.n_hidden = n_hidden
            modules = []
            modules.append(nn.Linear(x_size+z_size, self.n_hidden))
            modules.append(nn.ReLU())
            for i in range(2):
                modules.append(nn.Linear(self.n_hidden, self.n_hidden))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.n_hidden, 1))
            self.model = nn.Sequential(*tuple(modules))
        
        def forward(self, x, z):
            out = self.model(torch.cat([x.view(x.size(0), -1),
                                        z.view(z.size(0), -1)], dim=-1))
            return out
        
    class g_decoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super(g_decoder, self).__init__()
            self.model = decoder(*args, **kwargs)
        def forward(self, common, residual, unique):
            x = sum([common, residual, unique])
            return self.model(x)
    
    encoder_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 4,
        'block_type'        : basic_block,
        'num_channels_list' : [N, N, N*2, N*4],
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    f_factor_inst = f_factor(**encoder_kwargs)
    encoder_output_size = f_factor_inst.output_shape[0]
    
    decoder_kwargs = {
        'input_shape'       : encoder_output_size,
        'output_shape'      : image_size,
        'num_conv_blocks'   : 4,
        'block_type'        : basic_block,
        'num_channels_list' : [N*4, N*2, N, 1],
        'output_transform'  : torch.sigmoid,
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_in'         : False,
        'upsample_mode'     : 'conv',
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    conv_stack_f_kwargs = {
        'in_channels'       : N*2,
        'out_channels'      : N,
        'num_blocks'        : 1,
        'block_type'        : basic_block,
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    conv_stack_g_kwargs = {
        'in_channels'       : N,
        'out_channels'      : N*2,
        'num_blocks'        : 1,
        'block_type'        : basic_block,
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    discriminator_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 4,
        'block_type'        : tiny_block,
        'num_channels_list' : [N, N, N*2, N*4],
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    bottleneck_size = (N,)+encoder_output_size[1:]
    print("DEBUG: bottleneck_size={}".format(bottleneck_size))
    vector_size = np.product(bottleneck_size)
    submodel = {
        'f_factor'          : f_factor_inst,
        'f_common'          : conv_stack(**conv_stack_f_kwargs),
        'f_residual'        : conv_stack(**conv_stack_f_kwargs),
        'f_unique'          : conv_stack(**conv_stack_f_kwargs),
        'g_common'          : conv_stack(**conv_stack_g_kwargs),
        'g_residual'        : conv_stack(**conv_stack_g_kwargs),
        'g_unique'          : conv_stack(**conv_stack_g_kwargs),
        'g_output'          : g_decoder(**decoder_kwargs),
        'disc_A'            : discriminator(**discriminator_kwargs),
        'disc_B'            : discriminator(**discriminator_kwargs),
        'mutual_information': mi_estimation_network(x_size=vector_size,
                                                    z_size=vector_size,
                                                    n_hidden=100)}
    
    model = segmentation_model(**submodel,
                               loss_seg=dice_loss(),
                               z_size=bottleneck_size,
                               z_constant=0,
                               **lambdas,
                               grad_penalty=None,
                               disc_clip_norm=None)
    
    return model
