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
            self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
            self.deconv1_bn = nn.BatchNorm2d(d*8)
            self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1)
            self.conv2_bn = nn.BatchNorm2d(d*4+1)
            self.dc2_1 = nn.Conv2d(d*8, 1, 3, 1, 2, dilation=2)
            self.conv3 = nn.Conv2d(d*4+1, d*2, 3, 1, 1)
            self.conv3_bn = nn.BatchNorm2d(d*2+2)
            self.dc3_1 = nn.Conv2d(d*4+1, 1, 3, 1, 2, dilation=2)
            self.dc3_2 = nn.Conv2d(d*4+1, 1, 3, 1, 4, dilation=4)
            self.conv4 = nn.Conv2d(d*2+2, d, 3, 1, 1)
            self.conv4_bn = nn.BatchNorm2d(d+3)
            self.dc4_1 = nn.Conv2d(d*2+2, 1, 3, 1, 2, dilation=2)
            self.dc4_2 = nn.Conv2d(d*2+2, 1, 3, 1, 4, dilation=4)
            self.dc4_3 = nn.Conv2d(d*2+2, 1, 3, 1, 8, dilation=8)
            self.conv5 = nn.Conv2d(d+3, 1, kernel_size=3, stride=1, padding=1)

        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, common, residual, unique):
            #x = sum([common, residual, unique])
            x = common
            x = F.relu(self.deconv1_bn(self.deconv1(x)))
            x = F.upsample(x, scale_factor=2)
            x = F.relu(self.conv2_bn(torch.cat([
                    adjust_to_size(
                        self.dc2_1(F.avg_pool2d(x, 4, 1, 2,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    self.conv2(x)], dim=1)
                   ))
            x = F.upsample(x, scale_factor=2)
            x = F.relu(self.conv3_bn(torch.cat([
                    adjust_to_size(
                        self.dc3_1(F.avg_pool2d(x, 4, 1, 2,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    adjust_to_size(
                        self.dc3_2(F.avg_pool2d(x, 8, 1, 4,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    self.conv3(x)], dim=1)
                   ))
            x = F.upsample(x, scale_factor=2)
            x = F.relu(self.conv4_bn(torch.cat([
                    adjust_to_size(
                        self.dc4_1(F.avg_pool2d(x, 4, 1, 2,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    adjust_to_size(
                        self.dc4_2(F.avg_pool2d(x, 8, 1, 4,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    adjust_to_size(
                        self.dc4_3(F.avg_pool2d(x, 16, 1, 8,
                                                count_include_pad=False)),
                        x.size()[2:]),
                    self.conv4(x)], dim=1)
                   ))
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
