from architectures import image2image

def build_model():
    disc_params = {
        'input_nc': 4,
        'ndf': 64,
        'n_layers_D': 3,
        'norm': 'instance',
        'which_model_netD': 'n_layers'
    }
    model = {
        'g_atob': image2image.define_G(4, 4, 64, 'resnet_9blocks', norm='instance'),
        'g_btoa': image2image.define_G(4, 4, 64, 'resnet_9blocks', norm='instance'),
        'd_a': image2image.define_D(**disc_params),
        'd_b': image2image.define_D(**disc_params)
    }
    return model
