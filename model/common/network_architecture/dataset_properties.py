# Generating dataset-specific  U-net architecture requires information about the dataset
import numpy as np
import torch
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet, Upsample, StackedConvLayers
from nnunet.network_architecture.initialization import InitWeights_He

from torch import nn
def get_dataset_properties(path=None):
     nnunet_architecture_config_files = {
         # plans
        'input_channels': 2,
        'base_num_features': 30,
        'num_classes': 2,
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