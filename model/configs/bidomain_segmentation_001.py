import torch
from torch import nn
import numpy as np
from fcn_maker.blocks import (basic_block,
                              batch_normalization,
                              get_initializer)
from fcn_maker.loss import dice_loss
from model.common import (encoder,
                          decoder)
from model.bidomain_segmentation import (segmentation_model,
                                         mine)


def build_model():
    #lambdas = {
        #'lambda_disc'       :1,
        #'lambda_x_id'       :10,
        #'lambda_z_id'       :1,
        #'lambda_const'      :1,
        #'lambda_cyc'        :0,
        #'lambda_mi'         :1,
        #'lambda_seg'        :1}
    lambdas = {
        'lambda_disc'       :0,
        'lambda_x_id'       :0,
        'lambda_z_id'       :0,
        'lambda_const'      :0,
        'lambda_cyc'        :0,
        'lambda_mi'         :0,
        'lambda_seg'        :1}
    
    encoder_kwargs = {
        'input_shape'       : (1, 100, 100),
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [20, 20, 30, 50, 100],
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_out'        : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    decoder_kwargs = {
        'input_shape'       : (50, 7, 7),
        'output_shape'      : (1, 100, 100),
        'num_conv_blocks'   : 5,
        'block_type'        : basic_block,
        'num_channels_list' : [50, 50, 30, 20, 1],
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'vector_in'         : False,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    conv_stack_kwargs = {
        'in_channels'       : 50,
        'out_channels'      : 50,
        'num_blocks'        : 4,
        'block_type'        : basic_block,
        'skip'              : True,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : 'ReLU',
        'ndim'              : 2}
    
    class conv_stack(nn.Module):
        def __init__(self, in_channels, out_channels, num_blocks, block_type,
                     skip=True, dropout=0., normalization=batch_normalization,
                     norm_kwargs=None, conv_padding=True,
                     init='kaiming_normal_', nonlinearity='ReLU', ndim=2):
            super(conv_stack, self).__init__()
            self.in_channels   = in_channels
            self.out_channels  = out_channels
            self.num_blocks    = num_blocks
            self.block_type    = block_type
            self.skip          = skip
            self.dropout       = dropout
            self.normalization = normalization
            self.norm_kwargs   = norm_kwargs if norm_kwargs is not None else {}
            self.conv_padding  = conv_padding
            self.nonlinearity  = nonlinearity
            self.ndim          = ndim
            self.init          = init
            assert num_blocks > 0
            block_kwargs = {'num_filters': self.out_channels,
                            'skip': self.skip,
                            'dropout': self.dropout,
                            'normalization': self.normalization,
                            'norm_kwargs': self.norm_kwargs,
                            'conv_padding': self.conv_padding,
                            'init': self.init,
                            'nonlinearity': self.nonlinearity,
                            'ndim': self.ndim}
            blocks = [block_type(in_channels=self.in_channels,
                                 **block_kwargs)]
            for i in range(num_blocks-1):
                blocks.append(block_type(in_channels=self.out_channels,
                                         **block_kwargs))
            self.blocks = nn.Sequential(*tuple(blocks))
        def forward(self, x):
            return self.blocks(x)
    
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
            out = self.model(torch.cat([x.view(x.size(0), -1),
                                        z.view(z.size(0), -1)], dim=-1))
            return out
        
    class f_factor(nn.Module):
        def __init__(self, *args, **kwargs):
            super(f_factor, self).__init__()
            self.model = encoder(*args, **kwargs)
        def forward(self, x):
            a, b = torch.chunk(self.model(x), 2, dim=1)
            if not a.is_contiguous():
                a = a.contiguous()
            if not b.is_contiguous():
                b = b.contiguous()
            return a, b
    
    bottleneck_size = (50, 7, 7)
    vector_size = np.product(bottleneck_size)
    submodel = {
        'f_factor'          : f_factor(**encoder_kwargs),
        'f_common'          : conv_stack(**conv_stack_kwargs),
        'f_residual'        : conv_stack(**conv_stack_kwargs),
        'f_unique'          : conv_stack(**conv_stack_kwargs),
        'g_common'          : conv_stack(**conv_stack_kwargs),
        'g_residual'        : conv_stack(**conv_stack_kwargs),
        'g_unique'          : conv_stack(**conv_stack_kwargs),
        'g_output'          : decoder(**decoder_kwargs),
        'disc_A'            : encoder(**encoder_kwargs),
        'disc_B'            : encoder(**encoder_kwargs),
        'mutual_information': mi_estimation_network(x_size=vector_size,
                                                    z_size=vector_size,
                                                    n_hidden=100)}
    
    model = segmentation_model(**submodel,
                               loss_segmentation=dice_loss(),
                               z_size=bottleneck_size,
                               z_constant=0,
                               **lambdas)
    
    return model
