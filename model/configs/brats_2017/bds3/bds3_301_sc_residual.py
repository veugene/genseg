# nnunet-like model inmplemented as bds3_106, using no new code imports.
#
# NOTE: no short skip (like nnunet).
#
# NOTE: skips are after conv, not after relu (unlike nnunet)


from collections import OrderedDict
import torch
from torch import nn
from torch.nn.utils import remove_spectral_norm
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import convolution_transpose
from fcn_maker.loss import dice_loss
from model.common.network.basic import (adjust_to_size,
                                        batch_normalization,
                                        basic_block,
                                        block_abstract,
                                        convolution,
                                        #conv_block,
                                        #do_upsample,
                                        get_initializer,
                                        get_nonlinearity,
                                        #get_output_shape,
                                        instance_normalization,
                                        layer_normalization,
                                        munit_discriminator,
                                        norm_nlin_conv,
                                        pool_block,
                                        recursive_spectral_norm,
                                        repeat_block,
                                        shortcut)
from model.common.losses import dist_ratio_mse_abs
from model.bd_segmentation import segmentation_model


def build_model(lambda_disc=3,
                lambda_x_id=50,
                lambda_z_id=1,
                lambda_f_id=0,
                lambda_cyc=50,
                lambda_seg=0.01,
                lambda_enforce_sum=None):
    N = 480 # Number of features at the bottleneck.
    n = 128 # Number of features to sample at the bottleneck.
    image_size = (4, 240, 120)
    
    # Rescale lambdas if a sum is enforced.
    lambda_scale = 1.
    if lambda_enforce_sum is not None:
        lambda_sum = ( lambda_disc
                      +lambda_x_id
                      +lambda_z_id
                      +lambda_f_id
                      +lambda_cyc
                      +lambda_seg)
        lambda_scale = lambda_enforce_sum/lambda_sum
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [32, 64, 128, 256, 480, 480],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'ndim'                : 2}
    encoder_instance = encoder(**encoder_kwargs)
    enc_out_shape = (480, 8, 8)
    
    decoder_common_kwargs = {
        'input_shape'         : (N-n,)+enc_out_shape[1:],
        'out_channels'        : image_size[0],
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [480, 480, 256, 128, 64, 32],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'conv',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'cat',
        'ndim'                : 2}
    
    decoder_residual_kwargs = {
        'input_shape'         : enc_out_shape,
        'out_channels'        : image_size[0],
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_channels_list'   : [480, 480, 256, 128, 64, 32],
        'num_classes'         : 1,
        'skip'                : False,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
        'norm_kwargs'         : None,
        'padding_mode'        : 'reflect',
        'kernel_size'         : 3,
        'init'                : 'kaiming_normal_',
        'upsample_mode'       : 'conv',
        'nonlinearity'        : lambda : nn.ReLU(inplace=True),
        'long_skip_merge_mode': 'cat',
        'ndim'                : 2}
    
    discriminator_kwargs = {
        'input_dim'           : image_size[0],
        'num_channels_list'   : [N//8, N//4, N//2, N],
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
        'disc_A'            : munit_discriminator(**discriminator_kwargs),
        'disc_B'            : munit_discriminator(**discriminator_kwargs)}
    for m in submodel.values():
        if m is None:
            continue
        recursive_spectral_norm(m)
    remove_spectral_norm(submodel['decoder_residual'].out_conv[1].conv.op)
    remove_spectral_norm(submodel['decoder_residual'].classifier.op)
    
    model = segmentation_model(**submodel,
                               shape_sample=z_shape,
                               loss_gan='hinge',
                               loss_seg=dice_loss([1,2,4]),
                               relativistic=False,
                               rng=np.random.RandomState(1234),
                               lambda_disc=lambda_disc*lambda_scale,
                               lambda_x_id=lambda_x_id*lambda_scale,
                               lambda_z_id=lambda_z_id*lambda_scale,
                               lambda_f_id=lambda_f_id*lambda_scale,
                               lambda_cyc=lambda_cyc*lambda_scale,
                               lambda_seg=lambda_seg*lambda_scale)
    #print(encoder_instance)
    #print(submodel['decoder_residual'])
    
    return OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc_A'],
                             model.separate_networks['disc_B']])),
        ))


class encoder(nn.Module):
    def __init__(self, input_shape, num_conv_blocks, block_type,
                 num_channels_list, skip=True, dropout=0.,
                 normalization=instance_normalization, norm_kwargs=None,
                 padding_mode='constant', kernel_size=3,
                 init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
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
        self.ndim = ndim
        self.in_channels = input_shape[0]
        
        '''
        Set up blocks.
        '''
        self.blocks = nn.ModuleList()
        conv = convolution(in_channels=self.in_channels,
                           out_channels=self.num_channels_list[0],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode=self.padding_mode,
                           init=self.init)
        self.blocks.append(conv)
        self.out_channels = self.num_channels_list[0]
        self.add_block(num_filters=self.num_channels_list[0], skip=skip)
        for i in range(1, self.num_conv_blocks-1):
            self.add_block(num_filters=self.num_channels_list[i],
                           stride=(2, 2),
                           skip=skip)
            self.add_block(num_filters=self.num_channels_list[i],
                           skip=skip)
        # Bottleneck.
        self.add_block(num_filters=self.num_channels_list[-1],
                       stride=(2, 1),
                       skip=skip)
        if normalization is not None:
            block = normalization(ndim=self.ndim,
                                  num_features=self.out_channels,
                                  **self.norm_kwargs)
            self.blocks.append(block)
        self.blocks.append(get_nonlinearity(self.nonlinearity))
        
    def add_block(self, num_filters, **kwargs):
        block_kwargs = {
            'in_channels': self.out_channels,
            'num_filters': num_filters,
            'dropout': self.dropout,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'padding_mode': self.padding_mode,
            'kernel_size': self.kernel_size,
            'init': self.init,
            'nonlinearity': self.nonlinearity,
            'ndim': self.ndim
        }
        block_kwargs.update(kwargs)
        block = self.block_type(**block_kwargs)
        self.blocks.append(block)
        self.out_channels = num_filters
            
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
                skips.append(out_prev)
        return out, skips


class switching_normalization(nn.Module):
    def __init__(self, normalization, *args, **kwargs):
        super(switching_normalization, self).__init__()
        self.norm0 = normalization(*args, **kwargs)
        self.norm1 = normalization(*args, **kwargs)
        self.mode = 0
    def set_mode(self, mode):
        assert mode in [0, 1]
        self.mode = mode
    def forward(self, x):
        if self.mode==0:
            return self.norm0(x)
        else:
            return self.norm1(x)
    

class decoder(nn.Module):
    def __init__(self, input_shape, out_channels, num_conv_blocks, block_type,
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
        valid_modes = [None, 'sum', 'skinny_cat', 'cat']
        if long_skip_merge_mode not in valid_modes:
            raise ValueError("`long_skip_merge_mode` must be one of {}."
                             "".format(", ".join(["\'{}\'".format(mode)
                                                  for mode in valid_modes])))
        
        self.input_shape = input_shape
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
        
        # Normalization switch (translation, segmentation modes).
        def normalization_switch(*args, **kwargs):
            return switching_normalization(*args,
                                           normalization=self.normalization,
                                           **kwargs)
        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.ups    = nn.ModuleList()
        self.out_channels = self.input_shape[0]
        self.add_block(num_filters=self.num_channels_list[0],
                       skip=skip,
                       normalization=None,
                       nonlinearity=None)
        for n in range(1, self.num_conv_blocks):
            if n == 1:
                stride = (2, 1)
            else:
                stride = (2, 2)
            self.ups.append(
                do_upsample(
                    mode=self.upsample_mode,
                    stride=stride,
                    ndim=self.ndim,
                    in_channels=self.out_channels,
                    out_channels=self.num_channels_list[n],
                    kernel_size=stride,
                    init=self.init
                )
            )
            self.out_channels = self.num_channels_list[n]
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
                self.out_channels += 1
            elif self.long_skip_merge_mode=='cat':
                self.out_channels *= 2
            else:
                pass
            self.add_block(
                num_filters=self.num_channels_list[n],
                skip=self.skip,
                normalization=normalization_switch,
            )
            if n != self.num_conv_blocks-1:
                self.add_block(
                    num_filters=self.num_channels_list[n],
                    skip=self.skip,
                    normalization=normalization_switch,
                )
            
        '''
        Final output - change number of channels.
        '''
        self.pre_conv = norm_nlin_conv(in_channels=self.out_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=self.kernel_size,
                                       init=self.init,
                                       normalization=normalization_switch,
                                       norm_kwargs=self.norm_kwargs,
                                       nonlinearity=self.nonlinearity,
                                       padding_mode=self.padding_mode)
        kwargs_out_conv = {
            'in_channels': self.out_channels,
            'out_channels': out_channels,
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
        self.out_channels = out_channels
        
        # Classifier for segmentation (mode 1).
        if self.num_classes is not None:
            self.classifier = convolution(
                in_channels=self.out_channels,
                out_channels=self.num_classes,
                kernel_size=1,
                ndim=self.ndim)
    
    def add_block(self, num_filters, **kwargs):
        block_kwargs = {
            'in_channels': self.out_channels,
            'num_filters': num_filters,
            'dropout': self.dropout,
            'normalization': self.normalization,
            'norm_kwargs': self.norm_kwargs,
            'padding_mode': self.padding_mode,
            'kernel_size': self.kernel_size,
            'init': self.init,
            'nonlinearity': self.nonlinearity,
            'ndim': self.ndim
        }
        block_kwargs.update(kwargs)
        block = self.block_type(**block_kwargs)
        self.blocks.append(block)
        self.out_channels = num_filters
        
    def forward(self, z, skip_info=None, mode=0):
        # Set mode (0: trans, 1: seg).
        assert mode in [0, 1]
        for m in self.modules():
            if isinstance(m, switching_normalization):
                m.set_mode(mode)
        
        # Compute output.
        if skip_info is not None and mode==0:
            skip_info = skip_info[::-1]
        out = self.blocks[0](z)
        for n in range(self.num_conv_blocks-1):
            out = self.ups[n](out)
            skip = skip_info[n]
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
            if not out.is_contiguous():
                out = out.contiguous()
            out = self.blocks[2*n+1](out)
            if n != self.num_conv_blocks-2:
                out = self.blocks[2*n+2](out)
        out = self.pre_conv(out)
        out = self.out_conv[mode](out)
        if mode==0:
            out = torch.tanh(out)
            return out, skip_info
        elif mode==1:
            out = self.classifier(out)
            if self.num_classes==1:
                out = torch.sigmoid(out)
            else:
                out = torch.softmax(out, dim=1)
            return out
        else:
            AssertionError()


class conv_block(block_abstract):
    """
    A single basic 3x3 convolution.
    """
    def __init__(self, in_channels, num_filters, skip=True, dropout=0.,
                 normalization=batch_normalization, norm_kwargs=None,
                 conv_padding=True, padding_mode='constant', kernel_size=3,
                 stride=1, init='kaiming_normal_', nonlinearity='ReLU',
                 ndim=2):
        super(conv_block, self).__init__(in_channels, num_filters,
                                         False, False)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.out_channels = num_filters
        self.skip = skip
        self.dropout = dropout
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.conv_padding = conv_padding
        self.padding_mode = padding_mode
        if not hasattr(kernel_size, '__len__'):
            kernel_size = (kernel_size,)*ndim
        if len(kernel_size) != ndim:
            raise ValueError(f"kernel_size must have {ndim} dims but "
                              "is {kernel_size}")
        self.kernel_size = kernel_size
        if not hasattr(stride, '__len__'):
            stride = (stride,)*ndim
        if len(stride) != ndim:
            raise ValueError(f"stride must have {ndim} dims but is {stride}")
        self.stride = stride
        self.init = init
        self.nonlinearity = nonlinearity
        self.ndim = ndim
        self.op = []
        if normalization is not None:
            self.op += [normalization(ndim=ndim,
                                      num_features=in_channels,
                                      **norm_kwargs)]
        self.op += [get_nonlinearity(nonlinearity)]
        #if conv_padding:
            ## For odd kernel sizes, equivalent to kernel_size//2.
            ## For even kernel sizes, two cases:
            ##    (1) [kernel_size//2-1, kernel_size//2] @ stride 1
            ##    (2) kernel_size//2 @ stride 2
            ## This way, even kernel sizes yield the same output size as
            ## odd kernel sizes. When subsampling, even kernel sizes allow
            ## possible downscaling without aliasing.
            #padding = []
            #for i in range(ndim):
                #padding.extend(
                    #[(kernel_size[i]-1)//2,
                     #(kernel_size[i]-int(stride[i]))//2]
                #)
        #else:
            #padding = 0
        if conv_padding:
            padding = []
            for k in kernel_size:
                padding.extend([k//2, k//2])
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
        self.op = nn.ModuleList(self.op)
        self.op_shortcut = None
        if skip:
            self.op_shortcut = shortcut(in_channels=in_channels,
                                        out_channels=num_filters,
                                        subsample=False,
                                        upsample=False,
                                        init=init,
                                        ndim=ndim)

    def forward(self, input):
        out = input
        for op in self.op:
            out = op(out)
        if self.skip:
            out = self.op_shortcut(input, out)
        return out


class do_upsample(torch.nn.Module):
    def __init__(self, mode, ndim, init=None, stride=2, **conv_kwargs):
        super(do_upsample, self).__init__()
        if mode=='repeat':
            self.op = torch.nn.Upsample(scale_factor=2)
        elif mode=='conv':
            self.op = convolution_transpose(ndim=ndim,
                                            stride=stride,
                                            init=init,
                                            **conv_kwargs)
            self.in_channels = self.op.in_channels
            self.out_channels = self.op.out_channels
        else:
            raise ValueError("Unrecognized upsample_mode: {}"
                             "".format(upsample_mode))
        self.mode = mode
        self.ndim = ndim
        self.init = init

    def forward(self, input):
        return self.op(input)
