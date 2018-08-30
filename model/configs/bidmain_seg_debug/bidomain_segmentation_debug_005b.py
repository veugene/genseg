import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from fcn_maker.blocks import (adjust_to_size,
                              basic_block,
                              tiny_block,
                              convolution,
                              get_initializer,
                              get_nonlinearity)
from fcn_maker.loss import dice_loss
from model.common import (batch_normalization,
                          instance_normalization)
from model.bidomain_segmentation import segmentation_model
from model.mine import mine


def build_model():
    N = 128
    #Q = 16
    image_size = (1, 30, 30)
    lambdas = {
        'lambda_disc'       : 1,
        'lambda_x_id'       : 0,
        'lambda_z_id'       : 0,
        'lambda_const'      : 0,
        'lambda_cyc'        : 0,
        'lambda_mi'         : 0,
        'lambda_seg'        : 0}
    
    class encoder(nn.Module):
        # initializers
        def __init__(self, d=128):
            super(encoder, self).__init__()
            self.conv1 = nn.Conv2d(1, d, 4, 1, 1)
            self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(d*2)
            self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(d*4)
            self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, d*8, 3, 1, 1)

        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, input):
            x = F.leaky_relu(self.conv1(input), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = F.sigmoid(self.conv5(x))

            return x
    
    ## G(z)
    #class decoder(nn.Module):
        ## initializers
        #def __init__(self, d=128):
            #super(decoder, self).__init__()
            #self.deconv1 = nn.ConvTranspose2d(d*Q, d*8, 4, 1, 0)
            #self.deconv1_bn = nn.BatchNorm2d(d*8)
            #self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
            #self.deconv2_bn = nn.BatchNorm2d(d*4)
            #self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
            #self.deconv3_bn = nn.BatchNorm2d(d*2)
            #self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            #self.deconv4_bn = nn.BatchNorm2d(d)
            #self.conv5 = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)

        ## weight_init
        #def weight_init(self, mean, std):
            #for m in self._modules:
                #normal_init(self._modules[m], mean, std)

        ## forward method
        #def forward(self, common, residual, unique):
            ##x = sum([common, residual, unique])
            #x = common
            #x = F.relu(self.deconv1_bn(self.deconv1(x)))
            #x = F.relu(self.deconv2_bn(self.deconv2(x)))
            #x = F.relu(self.deconv3_bn(self.deconv3(x)))
            #x = F.relu(self.deconv4_bn(self.deconv4(x)))
            #x = F.tanh(self.conv5(x))
            #x = adjust_to_size(x, (30, 30))

            #return x
    
    # G(z)
    class decoder(nn.Module):
        # initializers
        def __init__(self, d=128):
            super(decoder, self).__init__()
            self.deconv1 = nn.ConvTranspose2d(d, d*8, 4, 1, 0)
            self.deconv1_bn = nn.BatchNorm2d(d*8)
            self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
            self.conv2_bn = nn.BatchNorm2d(d*4)
            self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1)
            self.conv3_bn = nn.BatchNorm2d(d*2)
            self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
            self.conv4_bn = nn.BatchNorm2d(d)
            self.conv5 = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)

        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, common, residual, unique):
            #x = sum([common, residual, unique])
            x = common
            x = F.relu(self.deconv1_bn(self.deconv1(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv4_bn(self.conv4(x)))
            x = F.tanh(self.conv5(x))
            x = adjust_to_size(x, (30, 30))

            return x

    class discriminator(nn.Module):
        # initializers
        def __init__(self, d=128):
            super(discriminator, self).__init__()
            self.conv1 = nn.Conv2d(1, d, 4, 1, 1)
            self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(d*2)
            self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(d*4)
            self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, 1, 3, 1, 0)

        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, input):
            x = F.leaky_relu(self.conv1(input), 0.2)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = F.sigmoid(self.conv5(x))

            return x

    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
    
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
    
    class conv_stack(nn.Module):
        def __init__(self, in_channels, out_channels, num_blocks, block_type,
                     skip=True, dropout=0.,
                     normalization=instance_normalization, norm_kwargs=None,
                     conv_padding=True, init='kaiming_normal_', 
                     nonlinearity='ReLU', ndim=2):
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
        
    class g_decoder(nn.Module):
        def __init__(self, *args, **kwargs):
            super(g_decoder, self).__init__()
            self.model = decoder(*args, **kwargs)
        def forward(self, common, residual, unique):
            x = sum([common, residual, unique])
            return self.model(x)
    
    conv_stack_f_kwargs = {
        'in_channels'       : N*4,
        'out_channels'      : N,
        'num_blocks'        : 1,
        'block_type'        : basic_block,
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : batch_normalization,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    conv_stack_g_kwargs = {
        'in_channels'       : N,
        'out_channels'      : N*8,
        'num_blocks'        : 1,
        'block_type'        : tiny_block,
        'skip'              : False,
        'dropout'           : 0.,
        'normalization'     : None,
        'norm_kwargs'       : None,
        'conv_padding'      : True,
        'init'              : 'kaiming_normal_',
        'nonlinearity'      : lambda: nn.LeakyReLU(0.2),
        'ndim'              : 2}
    
    #class conv_expand(nn.Module):
        #def __init__(self, d):
            #super(conv_expand, self).__init__()
            #self.conv1 = nn.Conv2d(d, d*Q, 1, 1, 0)
            #self.conv1_bn = nn.BatchNorm2d(d*Q)
        #def forward(self, x):
            #x = F.relu(self.conv1_bn(self.conv1(x)))
            #return x
    
    f_factor_inst = f_factor(N)
    G = decoder(N)
    D_A = discriminator(N)
    D_B = discriminator(N)
    G.weight_init(mean=0.0, std=0.02)
    D_A.weight_init(mean=0.0, std=0.02)
    D_B.weight_init(mean=0.0, std=0.02)
    
    bottleneck_size = (N, 1, 1)
    vector_size = np.product(bottleneck_size)
    submodel = {
        'f_factor'          : f_factor_inst,
        'f_common'          : conv_stack(**conv_stack_f_kwargs),
        'f_residual'        : conv_stack(**conv_stack_f_kwargs),
        'f_unique'          : conv_stack(**conv_stack_f_kwargs),
        #'g_common'          : conv_expand(N),
        #'g_common'          : nn.Conv2d(N, N, 1, 1, 0),
        #'g_common'          : conv_stack(**conv_stack_g_kwargs),
        'g_common'          : lambda x:x,
        'g_residual'        : conv_stack(**conv_stack_g_kwargs),
        'g_unique'          : conv_stack(**conv_stack_g_kwargs),
        'g_output'          : G,
        'disc_A'            : D_A,
        'disc_B'            : D_B,
        'mutual_information': mi_estimation_network(x_size=vector_size,
                                                    z_size=vector_size,
                                                    n_hidden=100)}
    
    model = segmentation_model(**submodel,
                               loss_seg=dice_loss(),
                               z_size=bottleneck_size,
                               z_constant=0,
                               **lambdas,
                               grad_penalty=None,
                               disc_clip_norm=None)
    
    return model
