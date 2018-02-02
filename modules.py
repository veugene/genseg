from __future__ import (print_function,
                        division)
import numpy as np
import torch
from torch.autograd import Variable

class overlap_tile(torch.nn.Module):
    def __init__(self, input_patch_size, model):
        super(overlap_tile, self).__init__()
        self.input_patch_size = input_patch_size
        self.model = model
        self.output_patch_size = None
        self._use_gpu = False
        self._modules['model'] = model
        
    def cuda(self, *args, **kwargs):
        self._use_gpu = True
        super(overlap_tile, self).cuda(*args, **kwargs)
        
    def forward(self, input):
        ndim = len(self.input_patch_size)+2
        if input.ndimension() != ndim:
            raise ValueError("Using an input patch size of {}, expecting the "
                             "input to have {} dimensions. Got {} dimensions."
                             "".format(self.input_patch_size, ndim,
                                       input.ndimension()))
                                       
        # If not already computed, compute the output patch size.
        # HACK: this uses a forward pass.
        if self.output_patch_size is None:
            _in = Variable(torch.zeros(*tuple(self.input_patch_size)).float())
            if self._use_gpu:
                _in.cuda()
            _out = self.model(_in)
            self.output_patch_size = tuple(_out.size())
    
        # Compute the required size of the padded input image.
        lost = np.subtract(self.output_patch_size, self.input_patch_size)
        new_shape = list(input.size()[:2]) + list([input.size()[i+2]+lost[i]
                                                  for i in range(ndim-2)])
        for i in range(ndim):
            # Pytorch is very picky about which kinds of integers are used to
            # define a shape.
            new_shape[i] = int(new_shape[i])
                            
        # Create and fill padded input image.
        indices = [slice(None, None)]*ndim
        spatial_dims = range(2, ndim)
        for dim in spatial_dims:
            pad = lost[dim-2]//2
            indices[dim] = slice(pad, input.size()[dim]+pad)
        input_padded = Variable(torch.zeros(*new_shape).float())
        if self._use_gpu:
            input_padded.cuda()
        input_padded[indices] = input
        
        # Create output image buffer.
        output = Variable(torch.zeros(*tuple(input.size())).float())
        if self._use_gpu:
            output.cuda()
        
        # Iterator over indices in an image, for sourcing or placing patches.
        def index_iterator(image_size):
            def recursive_idx(dim, idx):
                for i in range(0, image_size[dim],
                                  self.output_patch_size[dim-2]):
                    idx[dim] = i
                    if dim < ndim-1:
                        for idx in recursive_idx(dim+1, idx):
                            yield idx
                    else:
                        yield idx
            iterator = recursive_idx(dim=2, idx=[None]*ndim)
            return iterator
            
        # Return list of slices for selecting a patch.
        def get_patch_slices(corner_idx, patch_size):
            indices = [slice(None, None)]*ndim
            for dim in range(ndim):
                if corner_idx[dim] is not None:
                    indices[dim] = slice(corner_idx[dim],
                                         corner_idx[dim]+patch_size[dim-2])
            return indices
            
        # Process input patch-wise.
        iter_input = index_iterator(input_padded.size())
        iter_output = index_iterator(output.size())
        for idx_i, idx_o in zip(iter_input, iter_output):
            sl_i = get_patch_slices(idx_i, self.input_patch_size)
            sl_o = get_patch_slices(idx_o, self.output_patch_size)
            input_patch = input_padded[sl_i]
            output_patch = self.model(input_patch)
            output[sl_o] = output_patch
            
        return output
 

if __name__=='__main__':
    class identity_module(torch.nn.Module):
        def forward(self, input):
            return input

    model = identity_module()
    tiler = overlap_tile(input_patch_size=(3,5), model=model)

    edge = 10
    im_input = np.reshape(range(edge**2), (1,1,edge,edge)).astype(np.float32)
    im_input = Variable(torch.from_numpy(im_input))
    im_output = tiler(im_input)

    if np.all(im_output==im_input):
        print("Test PASSED")
    else:
        print("Test FAILED")
