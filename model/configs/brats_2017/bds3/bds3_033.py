from collections import OrderedDict
import torch
from torch import nn
from torch.functional import F
from torch.nn.utils import spectral_norm
import numpy as np
from fcn_maker.model import assemble_resunet
from fcn_maker.loss import dice_loss
from model.common.network.basic import (AdaptiveInstanceNorm2d,
                                        adjust_to_size,
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
                                        recursive_spectral_norm,
                                        repeat_block)
from model.common.losses import dist_ratio_mse_abs
from model.bd_segmentation import segmentation_model


def build_model():
    NUM_COND_CLASSES = 155 # BRATS has 155 slices per brain.
    N = 512 # Number of features at the bottleneck.
    n = 128 # Number of features to sample at the bottleneck.
    image_size = (4, 240, 120)
    lambdas = {
        'lambda_disc'       : 3,
        'lambda_x_id'       : 50,
        'lambda_z_id'       : 1,
        'lambda_f_id'       : 0,
        'lambda_cyc'        : 50,
        'lambda_seg'        : 0.01}
    
    encoder_kwargs = {
        'input_shape'         : image_size,
        'num_conv_blocks'     : 6,
        'block_type'          : conv_block,
        'num_resblocks'       : 4,
        'num_channels_list'   : [N//32, N//16, N//8, N//4, N//2, N],
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : instance_normalization,
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
        'num_conv_blocks'     : 5,
        'block_type'          : conv_block,
        'num_resblocks'       : 4,
        'num_channels_list'   : [N, N//2, N//4, N//8, N//16, N//32],
        'num_cond_classes'    : NUM_COND_CLASSES,
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
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
        'num_conv_blocks'     : 5,
        'block_type'          : conv_block,
        'num_resblocks'       : 4,
        'num_channels_list'   : [N, N//2, N//4, N//8, N//16, N//32],
        'num_classes'         : 4,
        'num_cond_classes'    : NUM_COND_CLASSES,
        'mlp_dim'             : 256, 
        'embedding_dim'       : 32,
        'skip'                : True,
        'dropout'             : 0.,
        'normalization'       : layer_normalization,
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
        'num_channels_list'   : [N//8, N//4, N//2, N],
        'num_cond_classes'    : NUM_COND_CLASSES,
        'num_scales'          : 3,
        'normalization'       : layer_normalization,
        'norm_kwargs'         : None,
        'kernel_size'         : 3,
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
        'disc_A'            : multiscale_projection_discriminator(
                                                    **discriminator_kwargs),
        'disc_B'            : multiscale_projection_discriminator(
                                                    **discriminator_kwargs)}
    for m in submodel.values():
        if m is None:
            continue
        recursive_spectral_norm(m, types=(nn.Embedding,))
    
    model = segmentation_model(**submodel,
                               shape_sample=z_shape,
                               loss_gan='hinge',
                               loss_seg=multi_class_dice_loss([1,2,4]),
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
                 num_resblocks, num_channels_list, skip=True, dropout=0.,
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
        self.num_resblocks = num_resblocks
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
        assert(num_resblocks>0)
        resblock_shape = (self.num_channels_list[-1],)+shape[1:]
        resblocks = repeat_block(
            basic_block,
            in_channels=last_channels,
            num_filters=resblock_shape[0],
            repetitions=self.num_resblocks,
            skip=self.skip,
            dropout=self.dropout,
            subsample=False,
            upsample=False,
            upsample_mode='repeat',
            normalization=self.normalization,
            norm_kwargs=self.norm_kwargs,
            padding_mode=self.padding_mode,
            kernel_size=3,
            init=self.init,
            nonlinearity=self.nonlinearity,
            ndim=self.ndim)
        self.blocks.append(resblocks)
        shape = resblock_shape
        last_channels = resblock_shape[0]
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
                 num_resblocks, num_channels_list, num_cond_classes,
                 num_classes=None, mlp_dim=256, embedding_dim=32, skip=True,
                 dropout=0., normalization=layer_normalization,
                 norm_kwargs=None, padding_mode='constant', kernel_size=3,
                 upsample_mode='conv', init='kaiming_normal_',
                 nonlinearity='ReLU', long_skip_merge_mode=None,
                 output_list=False, ndim=2):
        super(decoder, self).__init__()
        
        # ndim must be only 2 or 3.
        if ndim not in [2, 3]:
            raise ValueError("`ndim` must be either 2 or 3")
        
        # num_channels should be specified once for every block.
        if len(num_channels_list)!=num_conv_blocks+1:
            raise ValueError("`num_channels_list` must have the same number "
                             "of entries as there are blocks+1.")
        
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
        self.num_resblocks = num_resblocks
        self.num_channels_list = num_channels_list
        self.num_cond_classes = num_cond_classes
        self.num_classes = num_classes
        self.mlp_dim = mlp_dim
        self.embedding_dim = embedding_dim
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
        self.output_list = output_list
        self.ndim = ndim
        
        self.in_channels  = self.input_shape[0]
        self.out_channels = self.output_shape[0]
        
        # Compute all intermediate conv shapes by working backward from the 
        # output shape.
        self._shapes = [self.output_shape,
                        (self.num_channels_list[-1],)+self.output_shape[1:]]
        for i in range(1, self.num_conv_blocks+1):
            shape_spatial = np.array(self._shapes[-1][1:])//2
            shape = (self.num_channels_list[-i],)+tuple(shape_spatial)
            self._shapes.append(shape)
        resblock_shape = (self.num_channels_list[0],)+self.input_shape[1:]
        self._shapes.append(resblock_shape)
        self._shapes.append(self.input_shape)
        self._shapes = self._shapes[::-1]
        
        '''
        Set up blocks.
        '''
        self.cats   = nn.ModuleList()
        self.blocks = nn.ModuleList()
        shape = self.input_shape
        last_channels = shape[0]
        assert(num_resblocks>0)
        resblocks = repeat_block(
            basic_block,
            in_channels=last_channels,
            num_filters=resblock_shape[0],
            repetitions=self.num_resblocks,
            skip=self.skip,
            dropout=self.dropout,
            subsample=False,
            upsample=False,
            upsample_mode='repeat',
            normalization=AdaptiveInstanceNorm2d,
            padding_mode=self.padding_mode,
            kernel_size=3,
            init=self.init,
            nonlinearity=self.nonlinearity,
            ndim=self.ndim)
        self.blocks.append(resblocks)
        shape = resblock_shape
        last_channels = resblock_shape[0]
        for n in range(1, self.num_conv_blocks+1):
            def _select(a, b=None):
                return a if n>0 else b
            block = self.block_type(
                in_channels=last_channels,
                num_filters=self.num_channels_list[n],
                upsample=True,
                upsample_mode=self.upsample_mode,
                skip=_select(self.skip, False),
                dropout=self.dropout,
                normalization=AdaptiveInstanceNorm2d,
                padding_mode=self.padding_mode,
                kernel_size=self.kernel_size,
                init=self.init,
                nonlinearity=_select(self.nonlinearity),
                ndim=self.ndim)
            self.blocks.append(block)
            shape = get_output_shape(block, shape)
            last_channels = self.num_channels_list[n]
            if   self.long_skip_merge_mode=='skinny_cat':
                cat = conv_block(in_channels=self.num_channels_list[n],
                                 num_filters=1,
                                 skip=False,
                                 normalization=AdaptiveInstanceNorm2d,
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
                                       normalization=AdaptiveInstanceNorm2d,
                                       nonlinearity=self.nonlinearity,
                                       padding_mode=self.padding_mode)
        kwargs_out_conv = {
            'in_channels': last_channels,
            'out_channels': output_shape[0],
            'kernel_size': 7,
            'normalization': self.normalization,    # NOTE: not AdaIN.
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
        
        # Count number of AdaIN parameters required.
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        
        # MLP to predict AdaIN parameters.
        mlp_input_size = np.product(self.input_shape)+self.embedding_dim
        self.mlp_self = mlp(n_layers=4,
                            n_input=mlp_input_size,
                            n_output=num_adain_params,
                            n_hidden=mlp_dim,
                            init=self.init)
        self.mlp_skip = mlp(n_layers=4,
                            n_input=mlp_input_size,
                            n_output=num_adain_params,
                            n_hidden=mlp_dim,
                            init=self.init)
        
        # Class-specific embedding.
        self.embedding = nn.Embedding(self.num_cond_classes,
                                      self.embedding_dim)
        
        
    def forward(self, z, class_info, skip_info=None, mode=0):
        # Mode is one of (0: trans, 1: seg).
        assert mode in [0, 1]
        
        # Assign AdaIN parameters.
        class_info = torch.Tensor(class_info).long().cuda()
        z_embedding = self.embedding(class_info)
        if mode==0:
            adain_params = self.mlp_self(torch.cat([z.view(z.size(0), -1),
                                                    z_embedding], dim=1))
        if mode==1:
            skip_info, adain_params = skip_info
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
        
        # Compute output.
        out_list = []
        out = z
        if skip_info is not None and mode==0:
            skip_info = skip_info[::-1]
        for n, block in enumerate(self.blocks):
            if (self.long_skip_merge_mode=='pool' and skip_info is not None
                                                  and n<len(skip_info)+1
                                                  and n>0):
                skip = skip_info[n-1]
                out = adjust_to_size(out, skip.size()[2:])
                out = block(out, unpool_indices=skip)
            elif (self.long_skip_merge_mode is not None
                                                  and skip_info is not None
                                                  and n<len(skip_info)+1
                                                  and n>0):
                skip = skip_info[n-1]
                out = block(out)
                out = adjust_to_size(out, skip.size()[2:])
                if   self.long_skip_merge_mode=='skinny_cat':
                    cat = self.cats[n-1]
                    out = torch.cat([out, cat(skip)], dim=1)
                elif self.long_skip_merge_mode=='cat':
                    out = torch.cat([out, skip], dim=1)
                elif self.long_skip_merge_mode=='sum':
                    out = out+skip
                else:
                    ValueError()
            else:
                out = block(out)
            if not out.is_contiguous():
                out = out.contiguous()
            out_list.append(out)
        out = self.pre_conv(out)
        out = self.out_conv[mode](out)
        out_list.append(out)
        if self.output_list:
            out = out_list
        if mode==0:
            out = torch.tanh(out)
            adain_params = self.mlp_skip(torch.cat([z.view(z.size(0), -1),
                                                    z_embedding], dim=1))
            return out, (skip_info, adain_params)
        elif mode==1:
            out = self.classifier(out)
            out = torch.sigmoid(out)
            return out
        else:
            AssertionError()


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


class multi_class_dice_loss(object):
    def __init__(self, target_class=1, mask_class=None):
        if not hasattr(target_class, '__len__'):
            target_class = [target_class]
        self.target_class = target_class
        self.mask_class = mask_class
        self._losses_single = []
        for c in [0]+target_class:
            # Dice loss for each class individually.
            self._losses_single.append(dice_loss(target_class=c,
                                                 mask_class=mask_class))
        # Dice loss for all classes, combined.
        self._loss_all = dice_loss(target_class=target_class, 
                                   mask_class=mask_class)
    
    def __call__(self, y_pred, y_true):
        loss = self._loss_all(1.-y_pred[:,0:1], y_true)
        for i, l in enumerate(self._losses_single):
            loss += l(y_pred[:,i:i+1].contiguous(), y_true)
        loss /= len(self._losses_single)+1
        return loss


class multiscale_projection_discriminator(nn.Module):
    def __init__(self, input_dim, num_channels_list, num_cond_classes,
                 num_scales=3, normalization=None, norm_kwargs=None,
                 kernel_size=5,
                 nonlinearity=lambda:nn.LeakyReLU(0.2, inplace=True),
                 padding_mode='reflect', init='kaiming_normal_'):
        super(multiscale_projection_discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_channels_list = num_channels_list
        self.num_cond_classes = num_cond_classes
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
        cnn_kwargs = {'input_dim': input_dim,
                      'num_channels_list': num_channels_list,
                      'num_cond_classes': num_cond_classes,
                      'normalization': normalization,
                      'norm_kwargs': norm_kwargs,
                      'kernel_size': kernel_size,
                      'nonlinearity': nonlinearity,
                      'padding_mode': padding_mode,
                      'init': init}
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(projection_cnn_discriminator(**cnn_kwargs))

    def forward(self, x, class_info):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x, class_info))
            x = self.downsample(x)
        return outputs


class projection_cnn_discriminator(nn.Module):
    def __init__(self, input_dim, num_channels_list, num_cond_classes,
                 normalization=None, norm_kwargs=None, kernel_size=5,
                 nonlinearity=lambda:nn.LeakyReLU(0.2, inplace=True),
                 padding_mode='reflect', init='kaiming_normal_'):
        super(projection_cnn_discriminator, self).__init__()
        self.input_dim = input_dim
        self.num_channels_list = num_channels_list
        self.num_cond_classes = num_cond_classes
        self.normalization = normalization
        self.norm_kwargs = norm_kwargs
        self.kernel_size = kernel_size
        self.nonlinearity = nonlinearity
        self.padding_mode = padding_mode
        self.init = init
        
        # CNN
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
        self.cnn = nn.Sequential(*cnn)
        
        # Embedding projection.
        self.embedding = nn.Embedding(self.num_cond_classes,
                                      self.num_channels_list[-1])
        self.linear = nn.Linear(in_features=self.num_channels_list[-1],
                                out_features=1)
        
    def forward(self, x, class_info):
        class_info = torch.Tensor(class_info).long().cuda()
        out = self.cnn(x)
        out = torch.sum(out, dim=(2,3))     # Global sum pooling.
        out = self.linear(out)
        out = torch.sum(self.embedding(class_info)*out, dim=1, keepdim=True)
        return out


class mlp(nn.Module):
    def __init__(self, n_layers, n_input, n_output, n_hidden=None,
                 nonlinearity='ReLU', init='kaiming_normal_'):
        super(mlp, self).__init__()
        assert(n_layers > 0)
        self.n_layers = n_layers
        self.n_input  = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.nonlinearity = nonlinearity
        self.init = init
        if n_hidden is None:
            self.n_hidden = n_output
        layers = []
        if n_layers > 1:
            layers =  [nn.Linear(in_features=n_input, out_features=n_hidden),
                       get_nonlinearity(self.nonlinearity)]
            layers += [nn.Linear(in_features=n_hidden, out_features=n_hidden),
                       get_nonlinearity(self.nonlinearity)]*max(0, n_layers-2)
            layers += [nn.Linear(in_features=n_hidden, out_features=n_output),
                       get_nonlinearity(self.nonlinearity)]
        else:
            layers = [nn.Linear(in_features=n_input, out_features=n_output),
                      get_nonlinearity(self.nonlinearity)]
        self.model = nn.Sequential(*tuple(layers))
        for layer in self.model:
            if isinstance(layer, nn.Linear) and init is not None:
                layer.weight.data = get_initializer(init)(layer.weight.data)
    def forward(self, x):
        return self.model(x)
