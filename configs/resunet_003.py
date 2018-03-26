from collections import OrderedDict
import torch
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

def build_model():
    model_kwargs = OrderedDict((
        ('in_channels', 4),
        ('num_classes', 4),
        ('num_init_blocks', 2),
        ('num_main_blocks', 4),
        ('main_block_depth', 1),
        ('init_num_filters', 32),
        ('dropout', 0.),
        ('main_block', basic_block),
        ('init_block', tiny_block),
        ('norm_kwargs', {'momentum': 0.1}),
        ('nonlinearity', (torch.nn.LeakyReLU, {'inplace': True})),
        ('ndim', 2)
    ))
    return assemble_resunet(**model_kwargs)
