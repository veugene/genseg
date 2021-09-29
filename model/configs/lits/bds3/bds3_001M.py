# like 106 for brats
# TODO: we need to turn off the classifier becuase nnunet is a segmenter, not a reconstructor (for mode 0 in residual decoder
# TODO: spectral loss is probably OK now, but needs refactoring due to previous comment
# TODO: normalization switches needs to be implemented as well in this (mode=0, mode=1)...

from collections import OrderedDict
import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet, StackedConvLayers, ConvDropoutNormNonlin, Upsample
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
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


def get_dataset_properties(path=None):
    nnunet_architecture_config_files = {
        # plans
        'input_channels': 1,
        'base_num_features': 30,
        'num_classes': 1,
        'num_pool': 7,
        # Static in nnunet
        'num_conv_per_stage': 2,
        'feat_map_mul_on_downscale': 2,
        'conv_op': nn.Conv2d,
        'norm_op': nn.InstanceNorm2d,
        'dropout_op': nn.Dropout2d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op_kwargs': {'p': 0, 'inplace': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        'deep_supervision': True,
        'dropout_in_localization': False,
        'final_nonlin': lambda x: x,
        'weightInitializer': InitWeights_He(1e-2),
        'pool_op_kernel_sizes': [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        'conv_kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
        'upscale_logits': False,
        'convolutional_pooling': True,
        'convolutional_upsampling': True,
        'max_num_features': None,
        'basic_block': ConvDropoutNormNonlin,
        'seg_output_use_bias': False,
    }
    return nnunet_architecture_config_files


def build_model(lambda_disc=3,
                lambda_x_id=50,
                lambda_z_id=1,
                lambda_f_id=0,
                lambda_cyc=50,
                lambda_seg=0.01,
                lambda_enforce_sum=None,
                mixed_precision=True):
    # this should be probably removed
    N = 512  # Number of features at the bottleneck.
    n = 128  # Number of features to sample at the bottleneck.
    image_size = (2, 480, 480)

    # Rescale lambdas if a sum is enforced.
    lambda_scale = 1.
    if lambda_enforce_sum is not None:
        lambda_sum = (lambda_disc
                      + lambda_x_id
                      + lambda_z_id
                      + lambda_f_id
                      + lambda_cyc
                      + lambda_seg)
        lambda_scale = lambda_enforce_sum / lambda_sum

    encoder_kwargs = {}
    encoder_instance = encoder(**get_dataset_properties())
    enc_out_shape = encoder_instance.final_num_features
    final_num_features = encoder_instance.final_num_features

    discriminator_kwargs = {
        'input_dim': 1,
        'num_channels_list': [N // 8, N // 4, N // 2, N],
        'num_scales': 3,
        'normalization': layer_normalization,
        'norm_kwargs': None,
        'kernel_size': 4,
        'nonlinearity': lambda: nn.LeakyReLU(0.2, inplace=True),
        'padding_mode': 'reflect',
        'init': 'kaiming_normal_'}

    # TODO we need to calculate the out shape of encoder (bottleneck (?)) manually.
    enc_out_shape = [8, 2, 2]

    z_shape = (n,) + tuple(enc_out_shape[1:])
    #print((("DEBUG: sample_shape={}".format(z_shape))

    decoder_common_kwargs = get_dataset_properties("")
    decoder_common_kwargs["num_classes"] = None

    decoder_residual_kwargs = get_dataset_properties("")


    submodel = {
        'encoder': encoder_instance,
        'decoder_common': decoder(352,
                                  encoder_instance.conv_blocks_context,
                                  **decoder_common_kwargs),
        'decoder_residual': decoder(final_num_features,
                                    encoder_instance.conv_blocks_context,
                                    **decoder_residual_kwargs),
        'segmenter': None,
        'mutual_information': None,
        'disc_A': munit_discriminator(**discriminator_kwargs),
        'disc_B': munit_discriminator(**discriminator_kwargs)}
    for m in submodel.values():
        if m is None:
            continue
        recursive_spectral_norm(m)
    # TODO: do we have spectral norm in decoder residual atm? (?)
    remove_spectral_norm(submodel['decoder_residual'].conv_blocks_localization[-1][-1].blocks[-1].conv)
    remove_spectral_norm(submodel['decoder_residual'].seg_outputs[-1])

    # If mixed precision mode, create the amp gradient scaler.
    scaler = None
    if mixed_precision:
        ##print((("DEBUG using mixed precision")
        scaler = torch.cuda.amp.GradScaler()

    model = segmentation_model(**submodel,
                               scaler=scaler,
                               shape_sample=z_shape,
                               loss_gan='hinge',
                               loss_seg=dice_loss(),
                               relativistic=False,
                               rng=np.random.RandomState(1234),
                               lambda_disc=lambda_disc * lambda_scale,
                               lambda_x_id=lambda_x_id * lambda_scale,
                               lambda_z_id=lambda_z_id * lambda_scale,
                               lambda_f_id=lambda_f_id * lambda_scale,
                               lambda_cyc=lambda_cyc * lambda_scale,
                               lambda_seg=lambda_seg * lambda_scale)

    out = OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc_A'],
                             model.separate_networks['disc_B']])),
    ))
    if mixed_precision:
        out['scaler'] = model.scaler
    return out


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
        if self.mode == 0:
            return self.norm0(x)
        else:
            return self.norm1(x)

class encoder(nn.Module):
    SPACING_FACTOR_BETWEEN_STAGES = 2
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_FILTERS_2D = 480
    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False
                 ):
        super(encoder, self).__init__()
        self.final_num_features = None
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        pool_op = nn.MaxPool2d
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3)] * (num_pool + 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        output_features = base_num_features
        input_features = input_channels

        self.conv_blocks_context = []
        self.td = []

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            self.final_num_features = output_features
        else:
            self.final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, self.final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(##print((_module_training_status

    def forward(self, x):
        skips = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)

        return x, skips


class decoder(nn.Module):
    SPACING_FACTOR_BETWEEN_STAGES = 2
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_FILTERS_2D = 480
    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, final_num_features, conv_blocks_context,
                 input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False
                 ):
        super(decoder, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes

        # we replace the argument from the plans as we need softmax/sigmoid as well as the tanh function for decoder.
        # self.final_nonlin = final_nonlin
        self.segm_nonlin_softmax = lambda x: F.softmax(x, 1)
        self.segm_nonlin_sigmoid = lambda x: F.sigmoid(x)
        self.segm_nonlin_tanh = lambda x: F.tanh(x)
        # TODO: replace as True, as nnunet architecture uses this.
        self._deep_supervision = False
        self.do_ds = False

        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        upsample_mode = 'bilinear'
        transpconv = nn.ConvTranspose2d
        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [(2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3)] * (num_pool + 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        for u in range(num_pool):
            # this should be taken from the encoder
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x, skip_info, mode=0):
        seg_outputs = []
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skip_info[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            if mode == 0:
                    seg_outputs.append(self.segm_nonlin_tanh(self.seg_outputs[u](x)))
            elif mode == 1:
                if self.num_classes == 1:
                    seg_outputs.append(self.segm_nonlin_sigmoid(self.seg_outputs[u](x)))
                else:
                    seg_outputs.append(self.segm_nonlin_softmax(self.seg_outputs[u](x)))

        #   DEEP SUPERVISION - LET'S REMOVE THAT FOR A WHILE
        #if self._deep_supervision and self.do_ds:
        #    return tuple([seg_outputs[-1]] + [i(j) for i, j in
        #                                      zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), skip_info

        if mode == 0:
            return seg_outputs[-1], skip_info
        elif mode == 1:
            return seg_outputs[-1]

class mi_estimation_network(nn.Module):
    def __init__(self, x_size, z_size, n_hidden):
        super(mi_estimation_network, self).__init__()
        self.x_size = x_size
        self.z_size = z_size
        self.n_hidden = n_hidden
        modules = []
        modules.append(nn.Linear(x_size + z_size, self.n_hidden))
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


