##
## Code adapted from https://github.com/tbung/pytorch-revnet
## commit : 7cfcd34fb07866338f5364058b424009e67fbd20
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fcn_maker.model import fcn
from fcn_maker.blocks import identity_block
from .blocks import (rev_block,
                     reversible_basic_block,
                     dilated_rev_block,
                     tiny_block,
                     memoryless_block_wrapper)


class revnet(nn.Module):
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
        self.activations = []
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
    
    def free(self):
        del self.activations[:]
        
        
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
                             

def dilated_fcn_hybrid(in_channels, num_blocks, filters, dilation,
                       num_downscale=0, patch_size=None, short_skip=True,
                       long_skip=True, long_skip_merge_mode='sum',
                       upsample_mode='repeat', dropout=0.,
                       norm_kwargs=None, init='kaiming_normal',
                       nonlinearity='ReLU', block_type=reversible_basic_block,
                       num_classes=None, ndim=2, verbose=False):
    """
    Implements an FCN with a bottleneck containing a reversible dilated
    fully convolutional network, with arbitrary dilations. Blocks on the
    downscaling and upscaling paths use the overlap-tile method on
    convolutions to reduce memory usage.
    
    Args:
        in_channels (int) : The number of input channels.
        num_blocks (int) : The number of blocks to stack.
        filters (list or int) : Number of convolution filters per block. Also 
            for each down/up-scaling operation and the pre/post-processors.
        dilation (list or int) : Convolutional kernel dilation per block.
        num_downscale (int) : The number of times to downscale the input
            (and upscale the output), with each scaling operation being done
            using a convolution, each applied with the overlap-tile method. 
            On downscaling, strided convolution is used. On upscaling, columns
            and rows are repeated (nearest neighbour interpolation) and the
            result is cleaned up with a regular convolution. Output tile size
            is set to the resolution that is achieved after all downscale
            operations.
        patch_size (tuple of int) : The input patch size to use with the
            overlap-tile strategy. If set to `None`, the overlap-tile strategy
            is not used.
        short_skip (bool) : Whether to use ResNet-like shortcut connections
            from the input of each block to its output. The inputs are summed
            with the outputs.
        long_skip (bool) : Whether to use long skip connections from the
            downward path to the upward path. These can either concatenate or 
            sum features across.
        long_skip_merge_mode (string) : Either 'sum' or 'concat' features
            across skip.
        upsample_mode (string) : Either 'repeat' or 'conv'. With 'repeat',
            rows and colums are repeated as in nearest neighbour interpolation.
            With 'conv', upscaling is done via transposed convolution.
        dropout (float) : Probability of dropout.
        norm_kwargs (dict): Keyword arguments to pass to batch norm layers.
            For batch normalization, default momentum is 0.9.
        init (string or function) : Convolutional kernel initializer.
        nonlinearity (string or function) : Nonlinearity.
        block_type (Module) : The block type to use.
        num_classes (int) : The number of classes to predict over. If set
            to `None`, no classifier will be used.
        ndim (int) : Number of spatial dimensions (1, 2 or 3).
        verbose (bool) : Whether to print messages about model structure 
            during construction.
    """
    if not hasattr(filters, '__len__'):
        filters = [filters]*(num_blocks+2 + 2*num_downscale)
    if not hasattr(dilation, '__len__'):
        dilation = [dilation]*num_blocks
        
    '''
    Make sure each block has a filters and dilation setting.
    '''
    if len(filters) != num_blocks+2 + 2*num_downscale:
        raise ValueError("`filters` must be passed as an integer or a "
                         "list with {} elements when `num_blocks` is {} "
                         "and num_downscale is {}. Instead, it was passed "
                         "as a list with {} elements."
                         "".format(num_blocks+2 + 2*num_downscale, 
                                   num_blocks,
                                   num_downscale,
                                   len(filters)))
    if len(dilation) != num_blocks:
        raise ValueError("`dilation` must be passed as an integer or a "
                         "list with {} elements when `num_blocks` is {}. "
                         "Instead, it was passed as a list with {} "
                         "elements."
                         "".format(num_blocks, num_blocks, len(dilation)))
                            
    '''
    ndim must be only 2 or 3.
    '''
    if ndim not in [2, 3]:
        raise ValueError("ndim must be either 2 or 3")
            
    '''
    If batch_normalization is used and norm_kwargs is not set, set default
    kwargs.
    '''
    if norm_kwargs is None:
        norm_kwargs = {'momentum': 0.1,
                       'affine': True}
    
    '''
    Constant kwargs passed to the init and main blocks.
    '''
    block_kwargs = {'dropout': dropout,
                    'norm_kwargs': norm_kwargs,
                    'upsample_mode': upsample_mode,
                    'nonlinearity': nonlinearity,
                    'init': init,
                    'ndim': ndim}    
    
    '''
    Assemble all necessary blocks.
    '''
    blocks_down = []
    blocks_across = []
    blocks_up = []
    
    # The first block is just a convolution.
    # No normalization or nonlinearity on input.
    kwargs = {'num_filters': filters[0],
              'skip': False,
              'nonlinearity': None,
              'dropout': dropout,
              'init': init,
              'ndim': ndim}
    if patch_size is not None:
        kwargs['input_patch_size'] = patch_size
    blocks_down.append((tiny_block, kwargs))

    # Down
    for i in range(num_downscale):
        kwargs = {'num_filters': filters[i+1],
                  'skip': short_skip,
                  'subsample': True}
        if patch_size is not None:
            kwargs['input_patch_size'] = patch_size
        kwargs.update(block_kwargs)
        blocks_down.append((tiny_block, kwargs))
        
    # Bottleneck, dilated revnet
    kwargs = {'num_filters': filters[num_downscale+1:-num_downscale-1],
              'dilation': dilation,
              'block_type': block_type,
              'num_blocks': len(dilation)}
    kwargs.update(block_kwargs)
    blocks_across.append((dilated_rev_block, kwargs))
    
    # Up
    for i in range(num_downscale):
        kwargs = {'num_filters': filters[-num_downscale-1+i],
                  'skip': short_skip,
                  'upsample': True}
        if patch_size is not None:
            kwargs['input_patch_size'] = patch_size
        kwargs.update(block_kwargs)
        blocks_up.append((tiny_block, kwargs))
        
    # The last block is just a convolution.
    # As requested, normalization and nonlinearity applied on input.
    kwargs = {'num_filters': filters[-1],
              'skip': False,
              'norm_kwargs': norm_kwargs,
              'nonlinearity': nonlinearity,
              'dropout': dropout,
              'init': init,
              'ndim': ndim}
    if patch_size is not None:
        kwargs['input_patch_size'] = patch_size
    blocks_up.append((tiny_block, kwargs))
    
    # Wrap down and up paths so that activations are recomputed on backprop,
    # rather than saved during the forward pass.
    blocks_down = [(memoryless_block_wrapper(blocks_down), {})]
    blocks_up = [(memoryless_block_wrapper(blocks_up), {})]
        
    '''
    Assemble model.
    '''
    blocks = blocks_down + blocks_across + blocks_up
    model = fcn(in_channels=in_channels,
                num_classes=num_classes,
                blocks=blocks,
                long_skip=long_skip,
                long_skip_merge_mode=long_skip_merge_mode,
                ndim=ndim,
                verbose=verbose)
    return model
