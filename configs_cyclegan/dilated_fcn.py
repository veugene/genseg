from architectures import image2image
from torch import nn

def build_model():
    disc_params = {
        'input_nc': 4,
        'ndf': 64,
        'n_layers_D': 3,
        'norm': 'instance',
        'which_model_netD': 'n_layers'
    }
    gen_kwargs = {
        'in_channels': 4,
        'C': 24,
        'norm_layer': nn.InstanceNorm2d
    }
    model = {
        'g_atob': image2image.DilatedFCN(**gen_kwargs),
        'g_btoa': image2image.DilatedFCN(**gen_kwargs),
        'd_a': image2image.define_D(**disc_params),
        'd_b': image2image.define_D(**disc_params)
    }
    return model
