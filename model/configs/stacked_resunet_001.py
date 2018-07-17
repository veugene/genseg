from collections import OrderedDict
import torch
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

def build_model():
    in_channels = 4
    num_classes = 4
    model_kwargs = OrderedDict((
        ('num_init_blocks', 2),
        ('num_main_blocks', 3),
        ('main_block_depth', 1),
        ('init_num_filters', 32),
        ('dropout', 0.),
        ('main_block', basic_block),
        ('init_block', tiny_block),
        ('norm_kwargs', {'momentum': 0.1}),
        ('nonlinearity', (torch.nn.LeakyReLU, {'inplace': True})),
        ('ndim', 2)
    ))
    
    class stack(torch.nn.Module):
        def __init__(self):
            super(stack, self).__init__()
            self.model1 = assemble_resunet(in_channels=in_channels,
                                           num_classes=None,
                                           **model_kwargs)
            out_channels = model_kwargs['init_num_filters']
            self.model2 = assemble_resunet(in_channels=out_channels,
                                           num_classes=num_classes,
                                           **model_kwargs)
            
        def forward(self, x):
            x = self.model1(x)
            x = self.model2(x)
            return x
        
    return stack()
