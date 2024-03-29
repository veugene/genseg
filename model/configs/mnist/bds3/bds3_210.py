from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm
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
                                        munit_discriminator,
                                        norm_nlin_conv,
                                        pool_block,
                                        recursive_spectral_norm,
                                        repeat_block)
from model.common.losses import dist_ratio_mse_abs
from model.bd_segmentation import segmentation_model


class PixelwiseNorm(nn.Module):
    def __init__(self, ndim, num_features):
        super(PixelwiseNorm, self).__init__()
    def forward(self, x, alpha=1e-7):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


def build_model():
    N = 512 # Number of features at the bottleneck.
    n = 128 # Number of features to sample at the bottleneck.
    image_size = (1, 48, 48)
    lambdas = {
        'lambda_disc'       : 3,
        'lambda_x_id'       : 50,
        'lambda_z_id'       : 1,
        'lambda_f_id'       : 0,
        'lambda_cyc'        : 50,
        'lambda_seg'        : 0.01}
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 5,
        'block_type'          : conv_block,
        'num_channels_list'   : [N//16, N//8, N//4, N//2, N],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : PixelwiseNorm,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'skip_pool_indices'   : False,
        'ndim'                : 2}
    encoder_instance = encoder(**encoder_kwargs)
    enc_out_shape = encoder_instance.output_shape
    
    decoder_common_kwargs = {
        'input_shape'         : (N-n,)+enc_out_shape[1:],
        'output_shape'        : image_size,
        'num_conv_blocks'     : 4,
        'block_type'          : conv_block,
        'num_channels_list'   : [N//2, N//4, N//8, N//16],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : PixelwiseNorm,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'skinny_cat',
        'ndim'                : 2}
    
    decoder_residual_kwargs = {
        'input_shape'         : enc_out_shape,
        'output_shape'        : image_size,
        'num_conv_blocks'     : 4,
        'block_type'          : conv_block,
        'num_channels_list'   : [N//2, N//4, N//8, N//16],
        'num_classes'         : 1,
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : PixelwiseNorm,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'repeat',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'skinny_cat',
        'ndim'                : 2}
    
    discriminator_kwargs = {
        'input_dim'           : image_size[0],
        'num_channels_list'   : [N//4, N//2, N],
        'num_scales'          : 3,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'kernel_size'         : 4,
        'nonlinearity'        : lambda : nn.LeakyReLU(0.2, inplace=True),
        'padding_mode'        : 'reflect',
        'init'                : 'kaiming_normal_'}
    
    x_shape = (N-n,)+tuple(enc_out_shape[1:])
    z_shape = (n,)+tuple(enc_out_shape[1:])
    print("DEBUG: sample_shape={}".format(z_shape))
    submodel = {
        'encoder'           : encoder_instance,
        'decoder_common'    : decoder(**decoder_common_kwargs),
        'decoder_residual'  : decoder(**decoder_residual_kwargs),
        'segmenter'         : None,
        'mutual_information': mi_estimation_network(
                                            x_size=np.product(x_shape),
                                            z_size=np.product(z_shape),
                                            n_hidden=1000),
        'disc_A'            : munit_discriminator(**discriminator_kwargs),
        'disc_B'            : munit_discriminator(**discriminator_kwargs)}
    for m in submodel.values():
        if m is None:
            continue
    recursive_spectral_norm(submodel['disc_A'])
    recursive_spectral_norm(submodel['disc_B'])
    
    model = segmentation_model(**submodel,
                               shape_sample=z_shape,
                               loss_gan='hinge',
                               loss_seg=dice_loss(),
                               relativistic=False,
                               rng=np.random.RandomState(1234),
                               **lambdas)
    
    return OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc_A'],
                             model.separate_networks['disc_B']])),
        ('E', model.separate_networks['mi_estimator'])
        ))


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU',
                 skip_pool_indices=False, ndim=2):
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
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.init = init
        self.nonlinearity = nonlinearity
        self.skip_pool_indices = skip_pool_indices
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
            if isinstance(m, pool_block):
                out, indices = m(out_prev)
            else:
                out = m(out_prev)
            out_size = out.size()
            if out_size[-1]!=size[-1] or out_size[-2]!=size[-2]:
                # Skip forward feature stacks prior to resolution change.
                size = out_size
                if self.skip_pool_indices:
                    skips.append(indices)
                else:
                    skips.append(out_prev)
        return out, skips
    

class decoder(nn.Module):
    def __init__(self, input_shape, output_shape, num_conv_blocks, block_type,
                 num_channels_list, num_classes=None, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3, upsample_mode='conv',
                 init='kaiming_normal_', nonlinearity='ReLU',
                 long_skip_merge_mode=None, ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks.")
        
        # long_skip_merge_mode settings.
        valid_modes = [None, 'skinny_cat', 'cat', 'pool']
        if long_skip_merge_mode not in valid_modes:
            raise ValueError("`long_skip_merge_mode` must be one of {}."
                             "".format(", ".join(["\'{}\'".format(mode)
                                                  for mode in valid_modes])))
        
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
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.init = init
        self.nonlinearity = nonlinearity
        self.long_skip_merge_mode = long_skip_merge_mode
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.guides = nn.ModuleList()
        shape = self.input_shape
        last_channels = shape[0]
        for n in range(self.num_conv_blocks):
            def _select(a, b=None):
                return a if n>0 else b
            block = self.block_type(
                in_channels=last_channels,
                num_filters=self.num_channels_list[n],
                upsample=True,
                upsample_mode=self.upsample_mode,
                skip=_select(self.skip, False),
                dropout=self.dropout,
                normalization=_select(normalization),
                padding_mode=self.padding_mode,
                kernel_size=self.kernel_size,
                init=self.init,
                nonlinearity=_select(self.nonlinearity),
                ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            self.guides.append(guide_block(*shape))
            last_channels = self.num_channels_list[n]
            if   self.long_skip_merge_mode=='skinny_cat':
                cat = conv_block(in_channels=self.num_channels_list[n],
                                 num_filters=1,
                                 skip=False,
                                 normalization=instance_normalization,
                                 nonlinearity=None,
                                 kernel_size=1,
                                 init=self.init,
                                 ndim=self.ndim)
                self.cats.append(cat)
                last_channels += 1
            elif self.long_skip_merge_mode=='cat':
                last_channels *= 2
            else:
                pass
            
        '''
        Final output - change number of channels.
        '''
        self.pre_conv = norm_nlin_conv(in_channels=last_channels,
                                       out_channels=last_channels,  # NOTE
                                       kernel_size=self.kernel_size,
                                       init=self.init,              # NOTE
                                       normalization=normalization,
                                       norm_kwargs=self.norm_kwargs,
                                       nonlinearity=self.nonlinearity,
                                       padding_mode=self.padding_mode)
        kwargs_out_conv = {
            'in_channels': last_channels,
            'out_channels': output_shape[0],
            'kernel_size': 7,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'nonlinearity': self.nonlinearity,
            'padding_mode': self.padding_mode,
            'init': self.init,
            'ndim': self.ndim}
        self.out_conv = [norm_nlin_conv(**kwargs_out_conv),
                         norm_nlin_conv(**kwargs_out_conv)]     # 1 per mode.
        self.out_conv = nn.ModuleList(self.out_conv)
        
        # Classifier for segmentation (mode 1).
        if self.num_classes is not None:
            self.classifier = convolution(
                in_channels=self.out_channels,
                out_channels=self.num_classes,
                kernel_size=1,
                ndim=self.ndim)
        
        
    def forward(self, z, skip_info=None, mode=0):
        # Set mode (0: trans, 1: seg).
        assert mode in [0, 1]
        
        # Compute output.
        out = z
        if skip_info is not None and mode==0:
            skip_info = skip_info[::-1]
        for n, block in enumerate(self.blocks):
            if (self.long_skip_merge_mode=='pool' and skip_info is not None
                                                  and n<len(skip_info)):
                skip = skip_info[n]
                out = adjust_to_size(out, skip.size()[2:])
                out = block(out, unpool_indices=skip)
                if mode==1:
                    out = self.guides[n](out)
            elif (self.long_skip_merge_mode is not None
                                                  and skip_info is not None
                                                  and n<len(skip_info)):
                skip = skip_info[n]
                out = block(out)
                if mode==1:
                    out = self.guides[n](out)
                out = adjust_to_size(out, skip.size()[2:])
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = self.cats[n]
                    out = torch.cat([out, cat(skip)], dim=1)
                elif self.long_skip_merge_mode=='cat':
                    out = torch.cat([out, skip], dim=1)
                elif self.long_skip_merge_mode=='sum':
                    out = out+skip
                else:
                    ValueError()
            else:
                out = block(out)
                if mode==1:
                    out = self.guides[n](out)
            if not out.is_contiguous():
                out = out.contiguous()
        out = self.pre_conv(out)
        out = self.out_conv[mode](out)
        if mode==0:
            out = torch.tanh(out)
            return out, skip_info
        elif mode==1:
            out = self.classifier(out)
            out = torch.sigmoid(out)
            return out
        else:
            AssertionError()


class guide_block(nn.Module):
    def __init__(self, channels, width, height):
        super(guide_block, self).__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.s_c = nn.Parameter(torch.zeros(1,channels,1,1))
        self.s_w = nn.Parameter(torch.zeros(1,1,width,1))
        self.s_h = nn.Parameter(torch.zeros(1,1,1,height))
        self.b   = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x):
        return x*(1+self.s_c+self.s_w+self.s_h)+self.b


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
