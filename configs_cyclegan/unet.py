from architectures import image2image

def build_model():
    disc_params = {
        'input_nc': 4,
        'ndf': 64,
        'n_layers_D': 3,
        'norm': 'instance',
        'which_model_netD': 'n_layers'
    }
    g_params = {
        'input_nc': 4,
        'output_nc': 4,
        'ngf': 64,
        'norm': 'instance',
        'which_model_netG': 'unet_128'
    }
    model = {
        'g_atob': image2image.define_G(**g_params),
        'g_btoa': image2image.define_G(**g_params),
        'd_a': image2image.define_D(**disc_params),
        'd_b': image2image.define_D(**disc_params)
    }
    return model
