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
from model.common import (dist_ratio_mse_abs,
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
    
    class LayerNorm(nn.Module):
        def __init__(self, num_features, eps=1e-5, affine=True):
            super(LayerNorm, self).__init__()
            self.num_features = num_features
            self.affine = affine
            self.eps = eps

            if self.affine:
                self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
                self.beta = nn.Parameter(torch.zeros(num_features))

        def forward(self, x):
            shape = [-1] + [1] * (x.dim() - 1)
            # print(x.size())
            if x.size(0) == 1:
                # These two lines run much faster in pytorch 0.4 than the two lines listed below.
                mean = x.view(-1).mean().view(*shape)
                std = x.view(-1).std().view(*shape)
            else:
                mean = x.view(x.size(0), -1).mean(1).view(*shape)
                std = x.view(x.size(0), -1).std(1).view(*shape)

            x = (x - mean) / (std + self.eps)

            if self.affine:
                shape = [1, -1] + [1] * (x.dim() - 2)
                x = x * self.gamma.view(*shape) + self.beta.view(*shape)
            return x

    class ResBlock(nn.Module):
        def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
            super(ResBlock, self).__init__()

            model = []
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
            self.model = nn.Sequential(*model)

        def forward(self, x):
            residual = x
            out = self.model(x)
            out += residual
            return out
    
    class ResBlocks(nn.Module):
        def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
            super(ResBlocks, self).__init__()
            self.model = []
            for i in range(num_blocks):
                self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
            self.model = nn.Sequential(*self.model)

        def forward(self, x):
            return self.model(x)
    
    class Conv2dBlock(nn.Module):
        def __init__(self, input_dim ,output_dim, kernel_size, stride,
                    padding=0, norm='none', activation='relu', pad_type='zero', transpose=False):
            super(Conv2dBlock, self).__init__()
            self.use_bias = True
            # initialize padding
            if pad_type == 'reflect':
                self.pad = nn.ReflectionPad2d(padding)
            elif pad_type == 'replicate':
                self.pad = nn.ReplicationPad2d(padding)
            elif pad_type == 'zero':
                self.pad = nn.ZeroPad2d(padding)
            else:
                assert 0, "Unsupported padding type: {}".format(pad_type)

            # initialize normalization
            norm_dim = output_dim
            if norm == 'bn':
                self.norm = nn.BatchNorm2d(norm_dim)
            elif norm == 'in':
                #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
                self.norm = nn.InstanceNorm2d(norm_dim)
            elif norm == 'ln':
                self.norm = LayerNorm(norm_dim)
            elif norm == 'adain':
                self.norm = AdaptiveInstanceNorm2d(norm_dim)
            elif norm == 'none':
                self.norm = None
            else:
                assert 0, "Unsupported normalization: {}".format(norm)

            # initialize activation
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'lrelu':
                self.activation = nn.LeakyReLU(0.2, inplace=True)
            elif activation == 'prelu':
                self.activation = nn.PReLU()
            elif activation == 'selu':
                self.activation = nn.SELU(inplace=True)
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'none':
                self.activation = None
            else:
                assert 0, "Unsupported activation: {}".format(activation)

            # initialize convolution
            if transpose:
                conv_op = nn.ConvTranspose2d
            else:
                conv_op = nn.Conv2d
            self.conv = conv_op(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        def forward(self, x):
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            return x
    
    class LongSkipEncoder(nn.Module):
        def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, kernel_size_end=7, kernel_size=4):
            super(LongSkipEncoder, self).__init__()
            self.layers = []
            self.layers += [Conv2dBlock(input_dim, dim, kernel_size_end, 1, (kernel_size_end-1)//2, norm=norm, activation=activ, pad_type=pad_type)]
            # downsampling blocks
            _dim = dim
            for i in range(n_downsample):
                dim *= 2
                self.layers += [Conv2dBlock(_dim, dim, kernel_size, 2, (kernel_size-1)//2, norm=norm, activation=activ, pad_type=pad_type)]
                _dim = dim
            # residual blocks
            self.layers += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
            self.output_dim = dim
            self._model = nn.Sequential(*self.layers)

        def forward(self, x):
            outputs = []
            out = x
            for l in self.layers:
                out = l(out)
                outputs.append(out)
            return outputs

    class LongSkipDecoder(nn.Module):
        def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero',
                    kernel_size_end=7, kernel_size=5, merge_mode='sum'):
            super(LongSkipDecoder, self).__init__()

            self.merge_mode = merge_mode
            self.layers = []
            self.cat_layers = []
            # AdaIN residual blocks
            self.layers += [(ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type),)]
            # upsampling blocks
            def _prev_dim(dim):
                p = dim
                if self.merge_mode=='cat':
                    p *= 2
                elif self.merge_mode=='skinny_cat':
                    p += 1
                return p
            _dim = _prev_dim(dim)
            for i in range(n_upsample):
                dim //= 2
                if self.merge_mode=='skinny_cat':
                    self.cat_layers += [Conv2dBlock(_dim-1, 1, kernel_size=1, stride=1, padding=0, norm='none', activation='none')]
                self.layers += [(nn.Upsample(scale_factor=2),
                                Conv2dBlock(_dim, dim, kernel_size, 1, (kernel_size-1)//2, norm='ln', activation=activ, pad_type=pad_type))]
                _dim = _prev_dim(dim)
            # use reflection padding in the last conv layer
            if self.merge_mode=='skinny_cat':
                self.cat_layers += [Conv2dBlock(_dim-1, 1, kernel_size=1, stride=1, padding=0, norm='none', activation='none')]
            self.layers += [(Conv2dBlock(_dim, output_dim, kernel_size_end, 1, (kernel_size_end-1)//2, norm='none', activation='tanh', pad_type=pad_type),)]
            self._model = nn.Sequential(*[l for l_set in self.layers for l in l_set])
            self._model_cat = nn.Sequential(*self.cat_layers)

        def forward(self, x, skips):
            out = x
            for i, l_set in enumerate(self.layers):
                for l in l_set:
                    out = l(out)
                if i < len(self.layers)-1:
                    if self.merge_mode=='sum':
                        out = out+skips[-i-1]
                    elif self.merge_mode=='cat':
                        out = torch.cat([out, skips[-i-1]], dim=1)
                    elif self.merge_mode=='skinny_cat':
                        compressed_skip = self.cat_layers[i](skips[-i-1])
                        out = torch.cat([out, compressed_skip], dim=1)
                    elif self.merge_mode=='none':
                        pass
                    else:
                        raise ValueError()
            return out
    
    class decoder_join(nn.Module):
        def __init__(self, *args, **kwargs):
            super(decoder_join, self).__init__()
            self.model = LongSkipDecoder(*args, **kwargs)
        def forward(self, common, residual, unique, skips=None):
            x = torch.cat([common, residual, unique], dim=1)
            x = self.model(x, skips)
            return x
            
    class encoder_split(nn.Module):
        def __init__(self, *args, **kwargs):
            super(encoder_split, self).__init__()
            self.model = LongSkipEncoder(*args, **kwargs)
            self.output_dim = self.model.output_dim
        def forward(self, x):
            out = self.model(x)
            skips = out[:-1]
            a, r, b = torch.split(out[-1],
                                  [N-2*n, n, n], dim=1)
            if not a.is_contiguous():
                a = a.contiguous()
            if not r.is_contiguous():
                r = r.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
            return a, r, b, skips
    
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
    
    encoder_kwargs = {
        'input_dim'           : 1,
        'dim'                 : 32,
        'n_downsample'        : 4,
        'n_res'               : 4,
        'norm'                : 'in',
        'activ'               : 'relu',
        'pad_type'            : 'reflect',
        'kernel_size'         : 4,
        'kernel_size_end'     : 7}
    encoder_A = encoder_split(**encoder_kwargs)
    encoder_B = encoder_split(**encoder_kwargs)
    
    decoder_kwargs = {
        'output_dim'          : 1,
        'dim'                 : encoder_A.output_dim,
        'n_upsample'          : 4,
        'n_res'               : 4,
        'res_norm'            : 'bn',
        'activ'               : 'relu',
        'pad_type'            : 'reflect',
        'kernel_size'         : 5,
        'kernel_size_end'     : 7,
        'merge_mode'          : 'skinny_cat'}
    
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
    
    o = 48 // 2**4
    enc_out_shape = (N, o, o)
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
                              debug_no_constant=False,
                              z_size=sample_shape,
                              **lambdas)
    
    return model
