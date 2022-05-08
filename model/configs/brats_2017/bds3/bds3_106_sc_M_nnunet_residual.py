from collections import OrderedDict
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import ResidualLayer, BasicResidualBlock
from nnunet.network_architecture.generic_UNet import StackedConvLayers, ConvDropoutNormNonlin, Upsample
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
    props = {'conv_op': nn.Conv2d,
             'norm_op': nn.InstanceNorm2d,
             'dropout_op': nn.Dropout2d,
             'conv_op_kwargs': {'dilation': 1, 'bias': True},
             'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
             'dropout_op_kwargs': {'p': 0, 'inplace': True},
             'nonlin': nn.LeakyReLU,
             'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
             }

    nnunet_architecture_config_files = {
        # plans
        'input_channels': 4,
        'base_num_features': 32,
        'num_classes': 1,
        'props': props,
        'num_blocks_per_stage': [1, 1, 1, 1, 1],
        'feat_map_mul_on_downscale': 2,
        'pool_op_kernel_sizes': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
        'conv_kernel_sizes': [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
        'block': BasicResidualBlock,
    }
    return nnunet_architecture_config_files


def build_model(lambda_disc=3,
                lambda_x_id=50,
                lambda_z_id=1,
                lambda_f_id=0,
                lambda_cyc=50,
                lambda_seg=0.01,
                lambda_enforce_sum=None):
    N = 512 # Number of features at the bottleneck.
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
    

    encoder_instance = encoder(previous=None, **get_dataset_properties())
    enc_out_shape = [480, 15, 8]

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

    decoder_common_kwargs = get_dataset_properties("")
    decoder_common_kwargs["num_classes"] = None

    decoder_residual_kwargs = get_dataset_properties("")

    submodel = {
        'encoder'           : encoder_instance,
        'decoder_common'    : decoder(previous=encoder_instance, max_num_features=352,
                                  **decoder_common_kwargs),
        'decoder_residual'  : decoder(previous=encoder_instance, max_num_features=480,
                                    **decoder_residual_kwargs),
        'segmenter'         : None,
        'mutual_information': None,
        'disc_A'            : munit_discriminator(**discriminator_kwargs),
        'disc_B'            : munit_discriminator(**discriminator_kwargs)}

    for m in submodel.values():
        if m is None:
            continue
        recursive_spectral_norm(m)

    # remove_spectral_norm(submodel['decoder_residual'].stages[-1][-1][1])
    # remove_spectral_norm(submodel['decoder_residual'].stages[-1][-1][0])
    remove_spectral_norm(submodel['decoder_residual'].segmentation_output)
    
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

    print(model)
    
    return OrderedDict((
        ('G', model),
        ('D', nn.ModuleList([model.separate_networks['disc_A'],
                             model.separate_networks['disc_B']]))
        ))


class encoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, previous, num_classes, default_return_skips=True,
                 max_num_features=480, deep_supervision=False,
                 upscale_logits=False, block=BasicResidualBlock,
                 ):
        super(encoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])
        self.initial_norm = props['norm_op'](base_num_features, **props['norm_op_kwargs'])
        self.initial_nonlin = props['nonlin'](**props['nonlin_kwargs'])

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = ResidualLayer(current_input_features, current_output_features, current_kernel_size, props,
                                          self.num_blocks_per_stage[stage], current_pool_kernel_size, block)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=True):
        skips = []

        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x = s(x)
            skips.append(x)

        return skips[-1], skips[:-1][::-1]


class switching_normalization(nn.Module):
    def __init__(self, output_channels, **kwargs):
        super(switching_normalization, self).__init__()
        self.norm0 = nn.InstanceNorm2d(output_channels, **kwargs)
        self.norm1 = nn.InstanceNorm2d(output_channels, **kwargs)
        self.mode = 0

    def set_mode(self, mode):
        assert mode in [0, 1]
        self.mode = mode

    def forward(self, x):
        if self.mode == 0:
            return self.norm0(x)
        else:
            return self.norm1(x)

class decoder(nn.Module):
    def __init__(self, previous, num_classes, input_channels, base_num_features, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480,  num_blocks_per_stage=None, deep_supervision=False,
                 upscale_logits=False, block=BasicResidualBlock,
                 ):
        super(decoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.segm_nonlin_softmax = lambda x: F.softmax(x, 1)
        self.segm_nonlin_sigmoid = lambda x: F.sigmoid(x)
        self.segm_nonlin_tanh = lambda x: F.tanh(x)

        def normalization_switch(output_channels, **kwargs):
            return switching_normalization(
                output_channels,
                **kwargs)

        props["norm_op"] = normalization_switch

        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if props is None:
            self.props = previous.props
        else:
            self.props = props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        # assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the
        # bottleneck

        self.tus = []
        self.stages = []

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            if i == 0:
                features_below = max_num_features
            else:
                features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s + 1], bias=False))
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(ResidualLayer(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s],
                                             self.props, num_blocks_per_stage[i], None, block))

        props["conv_op_kwargs"]["kernel_size"] = 3
        props["conv_op_kwargs"]["padding"] = 1
        self.stages[-1] = nn.Sequential(
                              StackedConvLayers(features_skip*2,
                                                features_skip,
                                                2,
                                                props["conv_op"],
                                                props["conv_op_kwargs"],
                                                normalization_switch,
                                                props["norm_op_kwargs"],
                                                props["dropout_op"],
                                                props["dropout_op_kwargs"],
                                                props["nonlin"],
                                                props["nonlin_kwargs"],
                                                basic_block = ConvDropoutNormNonlin),
                              nn.ModuleList(
                                  [nn.Conv2d(features_skip, 4, 3, padding=1),
                                   nn.Conv2d(features_skip, 4, 3, padding=1)]
                               )
        )


        if num_classes is not None:
            self.segmentation_output = self.props['conv_op'](4, num_classes, 1, 1, 0, 1, 1, False)


        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, skip_info, mode=0):
        for m in self.modules():
            if isinstance(m, switching_normalization):
                m.set_mode(mode)

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = adjust_to_size(x, skip_info[i].size()[2:])
            x = torch.cat((x, skip_info[i]), dim=1)
            if i < len(self.tus) - 1:
                x = self.stages[i](x)
            else:
                x = self.stages[i][0](x)
                x = self.stages[i][1][mode](x)

        if mode == 0:
            return self.segm_nonlin_tanh(x), skip_info
        else:
            if self.num_classes == 1:
                return self.segm_nonlin_sigmoid(self.segmentation_output(x))
            else:
                return self.segm_nonlin_softmax(self.segmentation_output(x))
