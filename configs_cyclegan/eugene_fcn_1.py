from collections import OrderedDict
from torch import nn
import fcn_maker.blocks as fcn_blocks
import fcn_maker.model as fcn_model
from fcn_maker.model import assemble_resunet
from architectures import image2image

def build_generator():
    nf = 64
    model_kwargs = OrderedDict((
        ('num_classes', None),
        ('in_channels', 4),
        ('num_init_blocks', 3),
        ('num_main_blocks', 3),
        ('main_block_depth', [1,2,4,8,4,2,1]),
        ('init_num_filters', nf),
        ('short_skip', True),
        ('long_skip', True),
        ('long_skip_merge_mode', 'concat'),
        ('main_block', fcn_blocks.basic_block),
        ('init_block', fcn_blocks.tiny_block),
        #('skip_block', (fcn_blocks.scaleshift_skip, {})),
        ('upsample_mode', 'repeat'),
        ('dropout', 0.0),
        #('normalization', InstanceNormLayer),
        ('norm_kwargs', {}),
        #('init', 'HeNormal'),
        ('nonlinearity', 'LeakyReLU'),
        ('ndim', 2),
        ('verbose', True),
        ))
    model = fcn_model.assemble_resunet(**model_kwargs)
    model = nn.Sequential(model, nn.Conv2d(64, 4, 3, padding=1), nn.Tanh())
    return model

def build_model():
    disc_params = {
        'input_nc': 4,
        'ndf': 64,
        'n_layers_D': 3,
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
    build_model()
