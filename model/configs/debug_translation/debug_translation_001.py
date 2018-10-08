import torch
from torch import nn
import numpy as np
from fcn_maker.blocks import (basic_block,
                              tiny_block,
                              convolution,
                              norm_nlin_conv,
                              get_initializer,
                              get_nonlinearity)
from fcn_maker.loss import dice_loss
from model.common import (encoder,
                          decoder,
                          dist_ratio_mse_abs,
                          batch_normalization,
                          instance_normalization)
from model.translation_debug import translation_model


def build_model():
    N = 512 # Number of features at the bottleneck.
    n = 8   # Number of features (subset of N) for samples.
    image_size = (1, 50, 50)
    lambdas = {
        'lambda_disc'       : 1,
        'lambda_x_id'       : 10,
        'lambda_z_id'       : 1,
        'lambda_const'      : 0,
        'lambda_cyc'        : 0,
        'lambda_mi'         : 0}
    
    class encoder_split(nn.Module):
        def __init__(self, *args, **kwargs):
            super(encoder_split, self).__init__()
            self.model = encoder(*args, **kwargs)
            self.output_shape = self.model.output_shape
        def forward(self, x):
            out, skips = self.model(x)
            a, r, b = torch.split(out,
                                  [N-2*n, n, n], dim=1)
            if not a.is_contiguous():
                a = a.contiguous()
            if not r.is_contiguous():
                r = r.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
            return a, r, b, skips
        
    #class discriminator(nn.Module):
        #def __init__(self, *args, **kwargs):
            #super(discriminator, self).__init__()
            #self.encoder = encoder(*args, **kwargs)
            #self.nlin = get_nonlinearity(self.encoder.nonlinearity)
            #self.final_conv = convolution(
                #in_channels=self.encoder.out_channels,
                #out_channels=1,
                #kernel_size=1,
                #init=self.encoder.init)
            
        #def forward(self, x):
            #out, _ = self.encoder(x)
            #out = self.nlin(out)
            #out = self.final_conv(out)
            #out = torch.sigmoid(out)
            #return out
    
    class discriminator(nn.Module):
        def __init__(self, input_dim, num_channels_list, num_scales=3,
                     normalization=None, norm_kwargs=None, kernel_size=5,
                     nonlinearity=lambda:nn.LeakyReLU(0.2, inplace=True),
                     padding_mode='reflect', init='kaiming_normal_'):
            super(discriminator, self).__init__()
            self.input_dim = input_dim
            self.num_channels_list = num_channels_list
            self.num_scales = num_scales
            self.normalization = normalization
            self.norm_kwargs = norm_kwargs
            self.kernel_size = kernel_size
            self.nonlinearity = nonlinearity
            self.padding_mode = padding_mode
            self.init = init
            self.downsample = nn.AvgPool2d(3,
                                           stride=2,
                                           padding=[1, 1],
                                           count_include_pad=False)
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                self.cnns.append(self._make_net())

        def _make_net(self):
            cnn = []
            layer = convolution(in_channels=self.input_dim,
                                out_channels=self.num_channels_list[0],
                                kernel_size=self.kernel_size,
                                stride=2,
                                padding=self.kernel_size//2,
                                padding_mode=self.padding_mode,
                                init=self.init)
            cnn.append(layer)
            for ch0, ch1 in zip(self.num_channels_list[:-1],
                                self.num_channels_list[1:]):
                layer = norm_nlin_conv(in_channels=ch0,
                                       out_channels=ch1,
                                       kernel_size=self.kernel_size,
                                       subsample=True,
                                       conv_padding=True,
                                       padding_mode=self.padding_mode,
                                       init=self.init,
                                       nonlinearity=self.nonlinearity,
                                       normalization=self.normalization,
                                       norm_kwargs=self.norm_kwargs)
                cnn.append(layer)
            layer = convolution(in_channels=self.num_channels_list[-1],
                                out_channels=1,
                                kernel_size=1)
            cnn.append(layer)
            cnn = nn.Sequential(*cnn)
            return cnn

        def forward(self, x):
            outputs = []
            for model in self.cnns:
                outputs.append(torch.sigmoid(model(x)))
                x = self.downsample(x)
            out_flat = torch.cat([o.view(o.size(0), -1) for o in outputs],
                                 dim=1)
            return out_flat
    
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
        
    class decoder_join(nn.Module):
        def __init__(self, *args, **kwargs):
            super(decoder_join, self).__init__()
            self.model = decoder(*args, **kwargs)
        def forward(self, common, residual, unique, skips=None):
            x = torch.cat([common, residual, unique], dim=1)
            x = self.model(x, skip_info=skips)
            return x
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 5,
        'block_type'          : tiny_block,
        'num_channels_list'   : [N//16, N//8, N//4, N//2, N],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : batch_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    encoder_A = encoder_split(**encoder_kwargs)
    encoder_B = encoder_split(**encoder_kwargs)
    enc_out_shape = encoder_A.output_shape
    
    decoder_kwargs = {
        'input_shape'         : enc_out_shape,
        'output_shape'        : image_size,
        'num_conv_blocks'     : 5,
        'block_type'          : tiny_block,
        'num_channels_list'   : [N, N//2, N//4, N//8, N//16],
        'output_transform'    : torch.tanh,
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : batch_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 5,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'skinny_cat',
        'ndim'                : 2}
    
    #discriminator_kwargs = {
        #'input_shape'         : image_size,
        #'num_conv_blocks'     : 6,
        #'block_type'          : tiny_block,
        #'num_channels_list'   : [N//32, N//16, N//8, N//4, N//2, N],
        #'skip'                : False,
        #'dropout'             : 0.,
        #'normalization'       : batch_normalization,
        #'norm_kwargs'         : None,
        #'conv_padding'        : True,
        #'init'                : 'kaiming_normal_',
        #'nonlinearity'        : lambda  : nn.LeakyReLU(0.2),
        #'ndim'                : 2}
    
    discriminator_kwargs = {
        'input_dim'           : image_size[0],
        'num_channels_list'   : [N//32, N//16, N//8],
        'num_scales'          : 3,
        'normalization'       : None,
        'norm_kwargs'         : None,
        'kernel_size'         : 5,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.LeakyReLU(0.2, inplace=True),
        'padding_mode'        : 'reflect',
        'init'                : 'kaiming_normal_'}
    
    sample_shape = (n,)+enc_out_shape[1:]
    x_shape = (N-n,)+enc_out_shape[1:]
    z_shape = sample_shape
    print("DEBUG: sample_shape={}".format(sample_shape))
    submodel = {
        'encoder_A'           : encoder_A,
        'decoder_A'           : decoder_join(**decoder_kwargs),
        'encoder_B'           : encoder_B,
        'decoder_B'           : decoder_join(**decoder_kwargs),
        'disc_A'              : discriminator(**discriminator_kwargs),
        'disc_B'              : discriminator(**discriminator_kwargs),
        'mutual_information'  : mi_estimation_network(
                                            x_size=np.product(x_shape),
                                            z_size=np.product(z_shape),
                                            n_hidden=1000)}
    
    model = translation_model(**submodel,
                              #loss_rec=dist_ratio_mse_abs,
                              z_size=sample_shape,
                              **lambdas)
    
    return model
