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
    N = 1024
    n_latent = 80
    image_size = (1, 50, 50)
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
            x = torch.cat([common, residual, unique], dim=1)
            x = self.model(x)
            return x
    
    encoder_kwargs = {
        'input_shape'       : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [N//16, N//8, N//4, N//2, N],
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
    enc_out_shape = f_factor_inst.output_shape[0]
    
    decoder_kwargs = {
        'input_shape'       : (n_latent*3,)+enc_out_shape[1:],
        'output_shape'      : image_size,
        'num_conv_blocks'   : 5,
        'block_type'        : tiny_block,
        'num_channels_list' : [N//2, N//4, N//8, N//16, 1],
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
        'num_conv_blocks'   : 6,
        'block_type'        : tiny_block,
        'num_channels_list' : [N//16, N//16, N//8, N//4, N//2, N],
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    class conv_bn_relu(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(conv_bn_relu, self).__init__()
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
        def forward(self, x):
            return nn.functional.relu(self.bn(self.conv(x)))
    
    latent_shape = (n_latent,)+enc_out_shape[1:]
    print("DEBUG: latent_shape={}".format(latent_shape))
    vector_size = np.product(enc_out_shape)
    submodel = {
        'f_factor'          : f_factor_inst,
        'f_common'          : conv_bn_relu(enc_out_shape[0], n_latent),
        'f_residual'        : conv_bn_relu(enc_out_shape[0], n_latent),
        'f_unique'          : conv_bn_relu(enc_out_shape[0], n_latent),
        'g_common'          : lambda x:x,
        'g_residual'        : lambda x:x,
        'g_unique'          : lambda x:x,
        'g_output'          : g_decoder(**decoder_kwargs),
        'disc_A'            : discriminator(**discriminator_kwargs),
        'disc_B'            : discriminator(**discriminator_kwargs),
        'mutual_information': mi_estimation_network(x_size=vector_size,
                                                    z_size=vector_size,
                                                    n_hidden=100)}
    
    model = segmentation_model(**submodel,
                               loss_seg=dice_loss(),
                               z_size=latent_shape,
                               z_constant=0,
                               **lambdas)
    
    return model
