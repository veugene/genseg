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
        def __init__(self, d=128):
            super(generator, self).__init__()
            self.conv0 = nn.Conv2d(bottleneck_size[0], d*8, 3, 1, 1)
            self.conv0_bn = nn.BatchNorm2d(d*8)
            self.conv1 = nn.Conv2d(d*8, d*8, 3, 1, 1)
            self.conv1_bn = nn.BatchNorm2d(d*8)
            self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
            self.conv2_bn = nn.BatchNorm2d(d*4)
            self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1)
            self.conv3_bn = nn.BatchNorm2d(d*2)
            self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1)
            self.conv4_bn = nn.BatchNorm2d(d)
            self.conv5 = nn.Conv2d(d, d, 3, 1, 1)
            self.conv5_bn = nn.BatchNorm2d(d)
            self.conv6 = nn.Conv2d(d, 1, kernel_size=3, stride=1, padding=1)

        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, common, residual, unique):
            x = common
            x = F.relu(self.conv0_bn(self.conv0(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv4_bn(self.conv4(x)))
            x = F.relu(F.upsample(x, scale_factor=2))
            x = F.relu(self.conv5_bn(self.conv5(x)))
            x = F.tanh(self.conv6(x))
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
    
    G = generator(128)
    D = discriminator(128)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    
    submodel = {
        'g_common'          : lambda x: x,
        'g_residual'        : lambda x: x,
        'g_unique'          : lambda x: x,
        'g_output'          : G,
        'disc'              : D}
    model = gan(**submodel,
                z_size=bottleneck_size,
                z_constant=0)
    
    return model
