##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fcn_maker.blocks import convolution
from .blocks import (batch_normalization,
                     rev_block,
                     reversible_basic_block)


class base_model(nn.Module):
    """
    The base model for implementing a reversible network. This can be
    subclassed to create a custom model.
    
    Note the special care required in the forward() method when using
    rev_block objects: final rev_block activation must be stored in the 
    self.activations list.
    """
    def __init__(self):
        super(base_model, self).__init__()
        self.activations = []
        
    def forward(self):
        '''
        NOTE: The activation of the final call to a rev_block should be
              appended to self.activations, so that activations of all
              rev_block objects could be reconstructed from it druing
              backprop.
        '''
        raise NotImplementedError
        
    def cuda(self, device=None):
        for m in self.layers:
            m.cuda()
        return self._apply(lambda t: t.cuda(device))
    
    def cpu(self):
        for m in self.layers:
            m.cpu()
        return self._apply(lambda t: t.cpu())
    
    def free(self):
        del self.activations[:]


class revnet(base_model):
    def __init__(self, in_channels, units, filters, subsample, classes=None,
                 block_type=reversible_basic_block):
        """
        Implements a reversible ResNet.
        
        Args:
            in_channels (int) : The number of input channels.
            units (list) : Number of residual units in each group
            filters (list) : Number of filters in each unit including the
                inputlayer, so it is one item longer than units.
            subsample (list) : List of boolean values for each block,
                specifying whether it should do 2x spatial subsampling.
            block_type (Module) : The block type to use.
            classes (int) : The number of classes to predict over.
        """
        super(revnet, self).__init__()
        self.block = block_type
        self.layers = nn.ModuleList()

        # Input layers
        self.layers.append(nn.Conv2d(in_channels, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))
        self.layers.append(nn.ReLU())

        for i, group in enumerate(units):
            self.layers.append(self.block(filters[i], filters[i + 1],
                                          self.activations,
                                          subsample=subsample[i]))
            for unit in range(1, group):
                self.layers.append(self.block(filters[i + 1],
                                              filters[i + 1],
                                              self.activations))
        self.bn_last = nn.BatchNorm2d(filters[-1])
        self.fc = nn.Linear(filters[-1], classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self.activations.append(x)
        x = F.relu(self.bn_last(x))
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
        
def revnet38():
    model = revnet(units=[3, 3, 3],
                   filters=[32, 32, 64, 112],
                   subsample=[0, 1, 1],
                   classes=10)
    model.name = "revnet38"
    return model


def revnet110():
    model = revnet(units=[9, 9, 9],
                   filters=[32, 32, 64, 112],
                   subsample=[0, 1, 1],
                   classes=10)
    model.name = "revnet110"
    return model
