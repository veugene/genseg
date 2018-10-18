import torch
from torch import nn
from torch.functional import F
#from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.blocks import (adjust_to_size,
                              get_initializer,
                              get_nonlinearity,
                              shortcut,
                              do_upsample)
from fcn_maker.blocks import convolution as _convolution
from fcn_maker.loss import dice_loss
from model.common.network.basic import (get_output_shape,
                                        batch_normalization,
                                        instance_normalization,
                                        layer_normalization)
from model.common.network.basic import conv_block as _conv_block
from model.common.network.spectral_norm import spectral_norm
from model.common.losses import dist_ratio_mse_abs
from model.residual_bidomain_segmentation import segmentation_model


def build_model():
    N = 512 # Number of features at the bottleneck.
    n = 512 # Number of features to sample at the bottleneck.
    image_size = (4, 256, 128)
    lambdas = {
        'lambda_disc'       : 1,
        'lambda_x_id'       : 10,
        'lambda_z_id'       : 1,
        'lambda_cross'      : 1,
        'lambda_cyc'        : 1,
        'lambda_seg'        : 1,
        'lambda_sample'     : 0}
    
    encoder_kwargs = {
        'input_shape'         : (N//64,)+image_size[1:],
        'num_conv_blocks'     : 7,
        'block_type'          : conv_block,
        'num_channels_list'   : [N//64, N//32, N//16, N//8, N//4, N//2, N],
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 4,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    encoder_instance = encoder(**encoder_kwargs)
    enc_out_shape = encoder_instance.output_shape
    
    decoder_kwargs = {
        'input_shape'         : enc_out_shape,
        'output_shape'        : (N//64,)+image_size[1:],
        'num_conv_blocks'     : 7,
        'block_type'          : conv_block,
        'num_channels_list'   : [N, N//2, N//4, N//8, N//16, N//32, N//64],
        'num_classes'         : 1,
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'conv_padding'        : True,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 5,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    
    discriminator_kwargs = {
        'input_dim'           : image_size[0],
        'num_channels_list'   : [N//16, N//8, N//4],
        'num_scales'          : 4,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'kernel_size'         : 4,
        'nonlinearity'        : lambda : nn.LeakyReLU(0.2, inplace=True),
        'padding_mode'        : 'reflect',
        'init'                : 'kaiming_normal_'}
    
    shape_sample = (n,)+tuple(enc_out_shape[1:])
    print("DEBUG: sample_shape={}".format(shape_sample))
    submodel = {
        'encoder'           : encoder_instance,
        'decoder'           : decoder(**decoder_kwargs),
        'preprocessor'      : nn.Sequential(
                                  convolution(in_channels=4,
                                              out_channels=N//64,
                                              kernel_size=7,
                                              padding=3,
                                              padding_mode='reflect'),
                                  nn.ReLU()),
        'postprocessor'     : nn.Sequential(
                                  nn.ReLU(),
                                  convolution(in_channels=N//64,
                                              out_channels=4,
                                              kernel_size=7,
                                              padding=3,
                                              padding_mode='reflect'),
                                  nn.Tanh()),
        'disc_A'            : discriminator(**discriminator_kwargs),
        'disc_B'            : discriminator(**discriminator_kwargs),
        'disc_cross'        : discriminator(**discriminator_kwargs)}
    
    model = segmentation_model(**submodel,
                               shape_sample=shape_sample,
                               sample_image_space=False,
                               loss_gan='hinge',
                               loss_seg=dice_loss([4,5]),
                               relativistic=False,
                               rng=np.random.RandomState(1234),
                               **lambdas)
    
    return {'G' : model,
            'D' : nn.ModuleList(model.disc.values())}


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
                            padding=(self.kernel_size-1)//2,
                            padding_mode=self.padding_mode,
                            init=self.init)
        cnn.append(layer)
        for i, (ch0, ch1) in enumerate(zip(self.num_channels_list[:-1],
                                           self.num_channels_list[1:])):
            normalization = self.normalization if i>0 else None
            layer = norm_nlin_conv(in_channels=ch0,
                                   out_channels=ch1,
                                   kernel_size=self.kernel_size,
                                   subsample=True,
                                   conv_padding=True,
                                   padding_mode=self.padding_mode,
                                   init=self.init,
                                   nonlinearity=self.nonlinearity,
                                   normalization=normalization,
                                   norm_kwargs=self.norm_kwargs)
            cnn.append(layer)
        layer = norm_nlin_conv(in_channels=self.num_channels_list[-1],
                               out_channels=1,
                               kernel_size=1,
                               nonlinearity=self.nonlinearity,
                               normalization=self.normalization,
                               norm_kwargs=self.norm_kwargs)
        cnn.append(layer)
        cnn = nn.Sequential(*cnn)
        return cnn

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(encoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        self.input_shape = input_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels = input_shape[0]
        self.out_channels = self.num_channels_list[-1]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = self.in_channels
        conv = convolution(in_channels=last_channels,
                           out_channels=self.num_channels_list[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode=self.padding_mode,
                           init=self.init)
        self.blocks.append(conv)
        shape = get_output_shape(conv, shape)
        last_channels = self.num_channels_list[0]
        for i in range(1, self.num_conv_blocks):
            block = self.block_type(in_channels=last_channels,
                                    num_filters=self.num_channels_list[i],
                                    subsample=True,
                                    skip=skip,
                                    dropout=self.dropout,
                                    normalization=self.normalization,
                                    norm_kwargs=self.norm_kwargs,
                                    conv_padding=self.conv_padding,
                                    padding_mode=self.padding_mode,
                                    kernel_size=self.kernel_size,
                                    init=self.init,
                                    nonlinearity=self.nonlinearity,
                                    ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[i]
        if normalization is not None:
            block = normalization(ndim=self.ndim,
                                  num_features=last_channels,
                                  **self.norm_kwargs)
            self.blocks.append(block)
        self.blocks.append(get_nonlinearity(self.nonlinearity))
        self.output_shape = shape
            
    def forward(self, input):
        skips = []
        size = input.size()
        out = input
        for m in self.blocks:
            out_prev = out
            out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                skips.append(out_prev)
        return out, skips
    

class decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_conv_blocks, block_type,
                 num_channels_list, num_classes=None, skip=True, dropout=0.,
                 normalization=layer_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 upsample_mode='conv', init='kaiming_normal_',
                 nonlinearity='ReLU', ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_conv_blocks = num_conv_blocks
        self.block_type = block_type
        self.num_channels_list = num_channels_list
        self.num_classes = num_classes
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = {} if norm_kwargs is None else norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        # Compute all intermediate conv shapes by working backward from the 
        # output shape.
        self._shapes = [self.output_shape,
                        (self.num_channels_list[-1],)+self.output_shape[1:]]
        for i in range(2, self.num_conv_blocks):
            shape_spatial = np.array(self._shapes[-1][1:])//2
            shape = (self.num_channels_list[-i],)+tuple(shape_spatial)
            self._shapes.append(shape)
        self._shapes.append(self.input_shape)
        self._shapes = self._shapes[::-1]
        
        '''
        Set up blocks.
        '''
        self.cats_0   = nn.ModuleList()
        self.cats_1   = nn.ModuleList()
        self.blocks_0 = nn.ModuleList()
        self.blocks_1 = nn.ModuleList()
        shape = self.input_shape
        last_channels = shape[0]
        for n in range(self.num_conv_blocks):
            upsample = bool(n<self.num_conv_blocks-1)    # Not on last layer.
            def _select(a, b=None):
                return a if n>0 else b
            block_kwargs = {'in_channels': last_channels,
                            'num_filters': num_channels_list[n],
                            'upsample': upsample,
                            'upsample_mode': self.upsample_mode,
                            'skip': _select(self.skip, False),
                            'dropout': self.dropout,
                            'normalization': _select(self.normalization),
                            'conv_padding': self.conv_padding,
                            'padding_mode': self.padding_mode,
                            'kernel_size': self.kernel_size,
                            'init': self.init,
                            'nonlinearity': _select(self.nonlinearity),
                            'ndim': self.ndim}
            self.blocks_0.append(self.block_type(**block_kwargs))
            self.blocks_1.append(self.block_type(**block_kwargs))
            shape = get_output_shape(self.blocks_0[-1], shape)
            last_channels = self.num_channels_list[n]
            cat_kwargs = {'num_filters': 1,
                          'skip': False,
                          'normalization': instance_normalization,
                          'nonlinearity': None,
                          'kernel_size': 1,
                          'init': self.init,
                          'ndim': self.ndim}
            if upsample:
                self.cats_0.append(conv_block(in_channels=
                                              self.num_channels_list[n+1],
                                              **cat_kwargs))
                last_channels += 1
            self.cats_1.append(conv_block(self.num_channels_list[n],
                                          **cat_kwargs))
            
        '''
        Final output - change number of channels.
        '''
        out_kwargs = {'normalization': self.normalization,
                      'norm_kwargs': self.norm_kwargs,
                      'nonlinearity': self.nonlinearity,
                      'padding_mode': self.padding_mode}
        self.output_0 = norm_nlin_conv(in_channels=last_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       init=self.init,
                                       **out_kwargs)
        self.output_1 = nn.Sequential(
            norm_nlin_conv(in_channels=last_channels+1,
                           out_channels=self.out_channels,
                           kernel_size=3,
                           init=self.init,
                           **out_kwargs),
            norm_nlin_conv(in_channels=self.out_channels,
                           out_channels=self.out_channels,
                           kernel_size=7,
                           **out_kwargs),
            convolution(in_channels=self.out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1),
            nn.Sigmoid())
        
    def forward(self, z, skip_info=None, out_idx=0):
        out = out_1 = z
        if skip_info is not None:
            skip_info = skip_info[::-1]
        for n in range(len(self.blocks_0)):
            shape_in  = self._shapes[n]
            shape_out = self._shapes[n+1]
            spatial_shape_in = tuple(max(out.size(i+1),
                                         shape_out[i]-shape_in[i])
                                     for i in range(1, self.ndim+1))
            if np.any(np.less_equal(spatial_shape_in, 0)):
                spatial_shape_in = shape_in[1:]
            out = adjust_to_size(out, spatial_shape_in)
            
            if not out.is_contiguous():
                out = out.contiguous()
            out_0 = self.blocks_0[n](out)
            if out_idx==1:
                out_1 = self.blocks_1[n](out_1)
                out_1 = torch.cat([out_1, self.cats_1[n](out_0)], dim=1)
                out_1 = adjust_to_size(out_1, shape_out[1:])
            out = out_0
            out = adjust_to_size(out, shape_out[1:])
            if not out.is_contiguous():
                out = out.contiguous()
            if skip_info is not None and n<len(skip_info):
                skip = skip_info[n]
                out = torch.cat([out, self.cats_0[n](skip)], dim=1)
        if out_idx==0:
            out = self.output_0(out)
        elif out_idx==1 and self.num_classes:
            out = self.output_1(out_1)
        else:
            raise ValueError("Invalid `out_idx`.")
        return out


class conv_block(_conv_block):
    """
    A single basic 3x3 convolution.
    Unlike in tiny_block, stride instead of maxpool and upsample before conv.
    """
    def __init__(self, in_channels, num_filters, subsample=False,
                 upsample=False, upsample_mode='repeat', skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
        super(_conv_block, self).__init__(in_channels, num_filters,
                                         subsample, upsample)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.upsample_mode = upsample_mode
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        if upsample:
            self.op += [do_upsample(mode=upsample_mode,
                                    ndim=ndim,
                                    in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=2,
                                    init=init)]
        stride = 1
        if subsample:
            stride = 2
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self.op += [convolution(in_channels=in_channels,
                                out_channels=num_filters,
                                kernel_size=kernel_size,
                                ndim=ndim,
                                stride=stride,
                                init=init,
                                padding=padding,
                                padding_mode=padding_mode)]
        if dropout > 0:
            self.op += [get_dropout(dropout, nonlin=nonlinearity)]
        self._register_modules(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=subsample,
                                        upsample=upsample,
                                        upsample_mode=upsample_mode,
                                        init=init,
                                        ndim=ndim)
            self._register_modules({'shortcut': self.op_shortcut})

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


"""
Select 2D or 3D as argument (ndim) and initialize weights on creation.
"""
class convolution(_convolution):
    def __init__(self, ndim=2, init=None, padding=None,
                 padding_mode='constant', *args, **kwargs):
        super(_convolution, self).__init__()
        if ndim==2:
            conv = torch.nn.Conv2d
        elif ndim==3:
            conv = torch.nn.Conv3d
        else:
            ValueError("ndim must be 2 or 3")
        self.ndim = ndim
        self.init = init
        self.padding = padding
        self.padding_mode = padding_mode
        self.op = spectral_norm(conv(*args, **kwargs))
        self.in_channels = self.op.in_channels
        self.out_channels = self.op.out_channels
        if init is not None:
            self.op.weight.data = get_initializer(init)(self.op.weight.data)

    def forward(self, input):
        out = input
        if self.padding is not None:
            padding = self.padding
            if not hasattr(padding, '__len__'):
                padding = [self.padding]*self.ndim*2
            out = F.pad(out, pad=padding, mode=self.padding_mode, value=0)
        out = self.op(out)
        return out


"""
Helper to build a norm -> ReLU -> conv block
This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
"""
class norm_nlin_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 subsample=False, upsample=False, upsample_mode='repeat',
                 nonlinearity='ReLU', normalization=batch_normalization,
                 norm_kwargs=None, conv_padding=True, padding_mode='constant',
                 init='kaiming_normal_', ndim=2):
        super(norm_nlin_conv, self).__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subsample = subsample
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        self.init = init
        self.ndim = ndim
        if normalization is not None:
            self._modules['norm'] = normalization(ndim=ndim,
                                                  num_features=in_channels,
                                                  **norm_kwargs)
        self._modules['nlin'] = get_nonlinearity(nonlinearity)
        if upsample:
            self._modules['upsample'] = do_upsample(mode=upsample_mode,
                                                    ndim=ndim,
                                                    in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=2)
        stride = 1
        if subsample:
            stride = 2
        if conv_padding:
            # For odd kernel sizes, equivalent to kernel_size//2.
            # For even kernel sizes, two cases:
            #    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            #    (2) kernel_size//2 @ stride 2
            # This way, even kernel sizes yield the same output size as
            # odd kernel sizes. When subsampling, even kernel sizes allow
            # possible downscaling without aliasing.
            padding = [(kernel_size-1)//2,
                       (kernel_size-int(subsample))//2]*ndim
        else:
            padding = 0
        self._modules['conv'] = convolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            ndim=ndim,
                                            stride=stride,
                                            init=init,
                                            padding=padding,
                                            padding_mode=padding_mode)

    def forward(self, input):
        out = input
        for op in self._modules.values():
            out = op(out)
        return out