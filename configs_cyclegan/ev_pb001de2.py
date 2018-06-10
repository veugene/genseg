from collections import OrderedDict
import torch
from torch import nn
import fcn_maker.blocks as fcn_blocks
import fcn_maker.model as fcn_model
from fcn_maker.model import assemble_resunet
from architectures import image2image

def build_generator():
    model_kwargs = OrderedDict((
        ('in_channels', 4),
        ('num_classes', None),
        ('num_init_blocks', 1),
        ('num_main_blocks', 3),
        ('main_block_depth', [1,2,4,8,4,2,1]),
        ('init_num_filters', 32),
        ('dropout', 0.),
        ('main_block', fcn_blocks.basic_block),
        ('init_block', fcn_blocks.tiny_block),
        ('norm_kwargs', {'momentum': 0.1}),
        ('nonlinearity', (torch.nn.LeakyReLU, {'inplace': True})),
        ('ndim', 2)
    ))
    model = fcn_model.assemble_resunet(**model_kwargs)
    model = nn.Sequential(model, nn.Conv2d(32, 4, 3, padding=1), nn.Tanh())
    return model

def build_model():
    disc_params = {
        'input_nc': 4,
        'ndf': 4,
        'n_layers_D': 4,
        'norm': 'instance',
        'which_model_netD': 'n_layers'
    }
    model = {
        'g_atob': build_generator(),
        'g_btoa': build_generator(),
        'd_a': image2image.define_D(**disc_params),
        'd_b': image2image.define_D(**disc_params)
    }
    return model


if __name__ == '__main__':
    from util import count_params
    dd = build_model()
    for key in dd.keys():
        print(key, ":", count_params(dd[key]))
