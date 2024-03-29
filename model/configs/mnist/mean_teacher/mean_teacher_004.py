#  Semi Supervised Segmentation using Mean Teacher Approach
#  Implements: https://arxiv.org/abs/1807.04657

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
from model.mean_teacher_segmentation import segmentation_model


def build_model():
    model = segmentation_model(
        student=NoPoolNoBNASPP(),
        teacher=NoPoolNoBNASPP(),
        loss_seg=dice_loss(),
        lambda_con=10.,
        alpha_max=0.99)
    return {'G': model}


class NoPoolASPP(nn.Module):
    """
    Model from the paper:
    @article{arxiv1710.01269,
      author = {Christian S. Perone, Evan Calabrese, Julien Cohen-Adad},
      title = {Spinal cord gray matter segmentation using deep dilated convolutions},
      journal = {arXiv preprint arXiv:1710.01269},
      year = {2017}
    }
    """

    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(NoPoolASPP, self).__init__()

        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv2a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv2b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.branch1a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(32, 32, kernel_size=1)
        self.branch1b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(32, 32, kernel_size=3, padding=6, dilation=6)
        self.branch2a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(32, 32, kernel_size=3, padding=6, dilation=6)
        self.branch2b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(32, 32, kernel_size=3, padding=12, dilation=12)
        self.branch3a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(32, 32, kernel_size=3, padding=12, dilation=12)
        self.branch3b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(32, 32, kernel_size=3, padding=18, dilation=18)
        self.branch4a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(32, 32, kernel_size=3, padding=18, dilation=18)
        self.branch4b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(32, 32, kernel_size=3, padding=24, dilation=24)
        self.branch5a_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(32, 32, kernel_size=3, padding=24, dilation=24)
        self.branch5b_bn = nn.BatchNorm2d(32, momentum=bn_momentum)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)
        self.concat_bn = nn.BatchNorm2d(6 * 32, momentum=bn_momentum)

        self.amort = nn.Conv2d(6 * 32, 64, kernel_size=1)
        self.amort_bn = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(64, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = self.conv1a_bn(x)
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_bn(x)
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_bn(x)
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_bn(x)
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_bn(branch1)
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_bn(branch1)
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_bn(branch2)
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_bn(branch3)
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_bn(branch3)
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_bn(branch4)
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_bn(branch4)
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_bn(branch5)
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_bn(branch5)
        branch5 = self.branch5b_drop(branch5)

        # Global Average Pooling
        global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1,
                                   branch2,
                                   branch3,
                                   branch4,
                                   branch5,
                                   global_pool], dim=1)

        concatenation = self.concat_bn(concatenation)
        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_bn(amort)
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = F.sigmoid(predictions)

        return predictions


class NoPoolNoBNASPP(nn.Module):
    """
    Model from the paper:

    @article{arxiv1710.01269,
      author = {Christian S. Perone, Evan Calabrese, Julien Cohen-Adad},
      title = {Spinal cord gray matter segmentation using deep dilated convolutions},
      journal = {arXiv preprint arXiv:1710.01269},
      year = {2017}
    }
    """

    def __init__(self):

        super(NoPoolNoBNASPP, self).__init__()

        drop_rate = 0.4

        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1a_drop = nn.Dropout2d(drop_rate)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv1b_drop = nn.Dropout2d(drop_rate)

        self.conv2a = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv2a_drop = nn.Dropout2d(drop_rate)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv2b_drop = nn.Dropout2d(drop_rate)

        # Branch 1x1 convolution
        self.branch1a = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.branch1a_drop = nn.Dropout2d(drop_rate)
        self.branch1b = nn.Conv2d(32, 32, kernel_size=1)
        self.branch1b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 6
        self.branch2a = nn.Conv2d(32, 32, kernel_size=3, padding=6, dilation=6)
        self.branch2a_drop = nn.Dropout2d(drop_rate)
        self.branch2b = nn.Conv2d(32, 32, kernel_size=3, padding=6, dilation=6)
        self.branch2b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 12
        self.branch3a = nn.Conv2d(32, 32, kernel_size=3, padding=12, dilation=12)
        self.branch3a_drop = nn.Dropout2d(drop_rate)
        self.branch3b = nn.Conv2d(32, 32, kernel_size=3, padding=12, dilation=12)
        self.branch3b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 18
        self.branch4a = nn.Conv2d(32, 32, kernel_size=3, padding=18, dilation=18)
        self.branch4a_drop = nn.Dropout2d(drop_rate)
        self.branch4b = nn.Conv2d(32, 32, kernel_size=3, padding=18, dilation=18)
        self.branch4b_drop = nn.Dropout2d(drop_rate)

        # Branch for 3x3 rate 24
        self.branch5a = nn.Conv2d(32, 32, kernel_size=3, padding=24, dilation=24)
        self.branch5a_drop = nn.Dropout2d(drop_rate)
        self.branch5b = nn.Conv2d(32, 32, kernel_size=3, padding=24, dilation=24)
        self.branch5b_drop = nn.Dropout2d(drop_rate)

        self.concat_drop = nn.Dropout2d(drop_rate)

        self.amort = nn.Conv2d(5 * 32, 64, kernel_size=1)
        self.amort_drop = nn.Dropout2d(drop_rate)

        self.prediction = nn.Conv2d(64, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = self.conv1a_drop(x)

        x = F.relu(self.conv1b(x))
        x = self.conv1b_drop(x)

        x = F.relu(self.conv2a(x))
        x = self.conv2a_drop(x)
        x = F.relu(self.conv2b(x))
        x = self.conv2b_drop(x)

        # Branch 1x1 convolution
        branch1 = F.relu(self.branch1a(x))
        branch1 = self.branch1a_drop(branch1)
        branch1 = F.relu(self.branch1b(branch1))
        branch1 = self.branch1b_drop(branch1)

        # Branch for 3x3 rate 6
        branch2 = F.relu(self.branch2a(x))
        branch2 = self.branch2a_drop(branch2)
        branch2 = F.relu(self.branch2b(branch2))
        branch2 = self.branch2b_drop(branch2)

        # Branch for 3x3 rate 6
        branch3 = F.relu(self.branch3a(x))
        branch3 = self.branch3a_drop(branch3)
        branch3 = F.relu(self.branch3b(branch3))
        branch3 = self.branch3b_drop(branch3)

        # Branch for 3x3 rate 18
        branch4 = F.relu(self.branch4a(x))
        branch4 = self.branch4a_drop(branch4)
        branch4 = F.relu(self.branch4b(branch4))
        branch4 = self.branch4b_drop(branch4)

        # Branch for 3x3 rate 24
        branch5 = F.relu(self.branch5a(x))
        branch5 = self.branch5a_drop(branch5)
        branch5 = F.relu(self.branch5b(branch5))
        branch5 = self.branch5b_drop(branch5)

        ## Global Average Pooling
        #global_pool = F.avg_pool2d(x, kernel_size=x.size()[2:])
        #global_pool = global_pool.expand(x.size())

        concatenation = torch.cat([branch1,
                                   branch2,
                                   branch3,
                                   branch4,
                                   branch5], dim=1)

        concatenation = self.concat_drop(concatenation)

        amort = F.relu(self.amort(concatenation))
        amort = self.amort_drop(amort)

        predictions = self.prediction(amort)
        predictions = torch.sigmoid(predictions)

        return predictions






# UNET

class DownConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(out_feat, momentum=bn_momentum)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = self.up1(x)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Unet(nn.Module):
    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        # Downsampling path
        self.conv1 = DownConv(1, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        preds = torch.sigmoid(x11)

        return preds