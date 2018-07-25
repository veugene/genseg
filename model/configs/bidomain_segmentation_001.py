import torch
from torch import nn
from fcn_maker.blocks import (basic_block,
                              batch_normalization,
                              get_initializer)
from fcn_maker.loss import dice_loss
from model.common import (image_to_vector,
                          vector_to_image)
from model.bidomain_segmentation import (segmentation_model,
                                         mine)


def build_model():
    vector_size = 50
    
    lambdas = {
        'lambda_disc'       :1,
        'lambda_x_id'       :10,
        'lambda_z_id'       :1,
        'lambda_const'      :1,
        'lambda_cyc'        :0,
        'lambda_mi'         :1,
        'lambda_seg'        :1}
    
    image_to_vector_kwargs = {
        'input_shape'       : (1, 100, 100),
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [20, 30, 40, 50, 50, vector_size*2],
        'short_skip'        : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    vector_to_image_kwargs = {
        'input_len'         : vector_size,
        'output_shape'      : (1, 100, 100),
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [50, 50, 40, 30, 20, 1],
        'short_skip'        : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    vector_to_vector_kwargs = {
        'n_input'           : vector_size,
        'n_output'          : vector_size,
        'n_layers'          : 1,
        'n_hidden'          : 100,
        'init'              : 'kaiming_normal_'}
    
    class vector_to_vector(nn.Module):
        def __init__(self, n_input, n_output, n_layers, n_hidden=None,
                     init='kaiming_normal_'):
            super(vector_to_vector, self).__init__()
            assert(n_layers > 0)
            self.n_input  = n_input
            self.n_output = n_output
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            if n_hidden is None:
                self.n_hidden = n_output
            layers = []
            if n_layers > 1:
                layers =  [nn.Linear(in_features=n_input,
                                     out_features=n_hidden)]
                layers += [nn.Linear(in_features=n_hidden,
                                     out_features=n_hidden)]*max(0, n_layers-2)
                layers += [nn.Linear(in_features=n_hidden,
                                     out_features=n_output)]
            else:
                layers = [nn.Linear(in_features=n_input,
                                    out_features=n_output)]
            self.model = nn.Sequential(*tuple(layers))
            for layer in self.model:
                if isinstance(layer, nn.Linear) and init is not None:
                    layer.weight.data = get_initializer(init)(layer.weight.data)
        def forward(self, x):
            return self.model(x)
    
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
            out = self.model(torch.cat([x, z], dim=-1))
            return out
        
    class f_factor(nn.Module):
        def __init__(self, *args, **kwargs):
            super(f_factor, self).__init__()
            self.model = image_to_vector(*args, **kwargs)
        def forward(self, x):
            return torch.chunk(self.model(x), 2, dim=-1)
    
    submodel = {
        'f_factor'          : f_factor(**image_to_vector_kwargs),
        'f_common'          : vector_to_vector(**vector_to_vector_kwargs),
        'f_residual'        : vector_to_vector(**vector_to_vector_kwargs),
        'f_unique'          : vector_to_vector(**vector_to_vector_kwargs),
        'g_common'          : vector_to_vector(**vector_to_vector_kwargs),
        'g_residual'        : vector_to_vector(**vector_to_vector_kwargs),
        'g_unique'          : vector_to_vector(**vector_to_vector_kwargs),
        'g_output'          : vector_to_image(**vector_to_image_kwargs),
        'disc_A'            : image_to_vector(**image_to_vector_kwargs),
        'disc_B'            : image_to_vector(**image_to_vector_kwargs),
        'mutual_information': mi_estimation_network(x_size=vector_size,
                                                    z_size=vector_size,
                                                    n_hidden=100)}
    
    model = segmentation_model(**submodel,
                               loss_segmentation=dice_loss(),
                               z_size=vector_size,
                               z_constant=0)
    
    return model
