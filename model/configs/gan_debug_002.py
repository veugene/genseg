import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from fcn_maker.blocks import (basic_block,
                              tiny_block,
                              convolution,
                              get_initializer,
                              get_nonlinearity)
from fcn_maker.loss import dice_loss
from model.common import (encoder,
                          decoder,
                          instance_normalization)
from model.gan_debug import gan
from model.mine import mine


def build_model():
    image_size = (1, 30, 30)
    bottleneck_size = (100, 1, 1)
    
    # G(z)
    class generator(nn.Module):
        # initializers
        def __init__(self, input_size=100, n_class=30*30):
            super(generator, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(self.fc1.out_features, 512)
            self.fc3 = nn.Linear(self.fc2.out_features, 1024)
            self.fc4 = nn.Linear(self.fc3.out_features, n_class)

        # forward method
        def forward(self, common, residual, unique):
            x = F.leaky_relu(self.fc1(common.view(common.size(0), -1)), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = F.tanh(self.fc4(x))
            x = x.view(x.size(0),1,30,30)

            return x

    class discriminator(nn.Module):
        # initializers
        def __init__(self, input_size=30*30, n_class=1):
            super(discriminator, self).__init__()
            self.fc1 = nn.Linear(input_size, 1024)
            self.fc2 = nn.Linear(self.fc1.out_features, 512)
            self.fc3 = nn.Linear(self.fc2.out_features, 256)
            self.fc4 = nn.Linear(self.fc3.out_features, n_class)

        # forward method
        def forward(self, input):
            x = input.view(input.size(0), -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.dropout(x, 0.3)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.dropout(x, 0.3)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = F.dropout(x, 0.3)
            x = F.sigmoid(self.fc4(x))

            return x
    
    submodel = {
        'g_common'          : lambda x: x,
        'g_residual'        : lambda x: x,
        'g_unique'          : lambda x: x,
        'g_output'          : generator(),
        'disc'              : discriminator()}
    model = gan(**submodel,
                z_size=bottleneck_size,
                z_constant=0,
                grad_penalty=1.,
                disc_clip_norm=None)
    
    return model
