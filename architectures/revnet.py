##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fcn_maker.blocks import (convolution,
                              max_pooling)
from .blocks import (batch_normalization,
                     rev_block,
                     reversible_basic_block)
from .modules import overlap_tile


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
        self.free()
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


class dilated_fcn(base_model):
    def __init__(self, in_channels, num_blocks, filters, dilation,
                 num_downscale=0, overlap_tile_patch_size=None, dropout=0.,
                 init='kaiming_normal', nonlinearity='ReLU',
                 block_type=reversible_basic_block, classes=None, ndim=2):
        """
        Implements a reversible dilated fully convolutional network..
        
        Args:
            
            in_channels (int) : The number of input channels.
            num_blocks (int) : The number of blocks to stack.
            filters (list or int) : Number of convolution filters per block.
                Also for each down/up-scaling operation.
            dilation (list or int) : Convolutional kernel dilation per block.
            num_downscale (int) : The number of times to downscale the input
                (and upscale the output), with each scaling operation being
                done using a convolution, each applied with the overlap-tile
                method. On downscaling, strided convolution is used. On
                upscaling, columns and rows are repeated (nearest neighbour
                interpolation) and the result is cleaned up with a regular
                convolution. Output tile size is set to the resolution that is
                achieved after all downscale operations.
            overlap_tile_patch_size (tuple of int) : The input patch size to
                use with the overlap-tile strategy. If set to `None`, the
                overlap-tile strategy is not used.
            dropout (float) : Probability of dropout.
            init (string or function) : Convolutional kernel initializer.
            nonlinearity (string or function) : Nonlinearity.
            block_type (Module) : The block type to use.
            classes (int) : The number of classes to predict over. If set to
                `None`, no classifier will be used.
            ndim : Number of spatial dimensions (1, 2 or 3).
        """
        super(dilated_fcn, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        if hasattr(filters, '__len__'):
            self.filters = [i for i in filters]
        else:
            self.filters = [filters]*(num_blocks+1 + 2*num_downscale)
        if hasattr(dilation, '__len__'):
            self.dilation = [i for i in dilation]
        else:
            self.dilation = [dilation]*num_blocks
        self.num_downscale = num_downscale
        self.overlap_tile_patch_size = overlap_tile_patch_size
        self.dropout = dropout
        self.init = init
        self.nonlinearity = nonlinearity
        self.block_type = block_type
        self.ndim = ndim
        self.classes = classes
        
        if len(filters) != num_blocks+1 + 2*num_downscale:
            raise ValueError("`filters` must be passed as an integer or a "
                             "list with {} elements when `num_blocks` is {} "
                             "and num_downscale is {}. Instead, it was passed "
                             "as a list with {} elements."
                             "".format(num_blocks+1 + 2*num_downscale, 
                                       num_blocks,
                                       num_downscale,
                                       len(filters)))
        if len(dilation) != num_blocks:
            raise ValueError("`dilation` must be passed as an integer or a "
                             "list with {} elements when `num_blocks` is {}. "
                             "Instead, it was passed as a list with {} "
                             "elements."
                             "".format(num_blocks, num_blocks, len(dilation)))
                             
        # Build network
        self.layers_down = []
        self.layers_across = []
        self.layers_up = []
        padding = 1
        if overlap_tile_patch_size is not None:
            padding = 0
        preprocessor = convolution(in_channels=in_channels,
                                   out_channels=filters[0],
                                   kernel_size=3,
                                   padding=padding,
                                   init=init,
                                   ndim=ndim)
        if overlap_tile_patch_size is not None:
            preprocessor = overlap_tile(overlap_tile_patch_size,
                                        model=preprocessor,
                                        in_channels=preprocessor.in_channels,
                                        out_channels=preprocessor.out_channels)
        prev_channels = filters[0]
        self.layers_down.append(preprocessor)
        for i in range(num_downscale):
            padding = 1
            if overlap_tile_patch_size is not None:
                padding = 0
            conv_down = convolution(in_channels=prev_channels,
                                    out_channels=filters[i+1],
                                    kernel_size=3,
                                    padding=padding,
                                    init=init,
                                    ndim=ndim)
            if overlap_tile_patch_size is not None:
                conv_down = overlap_tile(overlap_tile_patch_size,
                                         model=conv_down,
                                         in_channels=conv_down.in_channels,
                                         out_channels=conv_down.out_channels)
            prev_channels = filters[i+1]
            self.layers_down.append(max_pooling(kernel_size=2, ndim=ndim))
            self.layers_down.append(conv_down)
        for i in range(num_blocks):
            block = block_type(in_channels=prev_channels,
                               out_channels=filters[i+num_downscale+1],
                               nonlinearity=nonlinearity,
                               dropout=dropout,
                               dilation=dilation[i],
                               init=init,
                               ndim=ndim,
                               activations=self.activations)
            prev_channels = filters[i+num_downscale+1]
            self.layers_across.append(block)
        for i in range(num_downscale):
            filters_i = filters[i+num_blocks+num_downscale+1]
            padding = 1
            if overlap_tile_patch_size is not None:
                padding = 0
            conv_up = convolution(in_channels=prev_channels,
                                  out_channels=filters_i,
                                  kernel_size=3,
                                  padding=padding,
                                  stride=1,
                                  init=init,
                                  ndim=ndim)
            if overlap_tile_patch_size is not None:
                conv_up = overlap_tile(overlap_tile_patch_size,
                                       model=conv_up,
                                       in_channels=conv_up.in_channels,
                                       out_channels=conv_up.out_channels)
                ## TODO: Memorize the output_size on the downscaling path
                ##       and crop outputs to that size since they may become
                ##       larger with some input sizes.
            prev_channels = filters_i
            self.layers_up.append(torch.nn.Upsample(scale_factor=ndim))
            self.layers_up.append(conv_up)
        self.classifier = None
        if classes is not None:
            self.classifier = convolution(in_channels=filters[-1],
                                          out_channels=classes,
                                          kernel_size=1,
                                          ndim=ndim)
            
        # Coalesce layer list.
        self.layers = nn.ModuleList()
        for l in self.layers_down:
            self.layers.append(l)
        for l in self.layers_across:
            self.layers.append(l)
        for l in self.layers_up:
            self.layers.append(l)
        
    def forward(self, x):
        self.free()
        for layer in self.layers_down:
            x = layer(x)
        for layer in self.layers_across:
            x = layer(x)
        self.activations.append(x)
        for layer in self.layers_up:
            x = layer(x)
        
        # Output
        if self.classifier is not None:
            x = self.classifier(x)
            if self.classes==1:
                x = F.sigmoid(x)
            else:
                # Softmax that works on ND inputs.
                e = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
                s = torch.sum(e, dim=1, keepdim=True)
                x = e / s
                
        return x
