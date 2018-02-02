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
        
    def cuda(self, device=None):
        self._use_gpu = True
        self._device = device
        return self._apply(lambda t: t.cuda(device))
        
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
            input_size = (1,1)+tuple(self.input_patch_size)
            _in = Variable(torch.zeros(*input_size).float())
            if self._use_gpu:
                _in = _in.cuda(self._device)
            _out = self.model(_in)
            self.output_patch_size = tuple(_out.size()[2:])
    
        # Compute the required size of the padded input image.
        lost = np.subtract(self.input_patch_size, self.output_patch_size)
        new_shape = list(input.size()[:2])
        for i in range(ndim-2):
            pad0 = lost[i]//2
            pad1 = self.input_patch_size[i] \
                   - (input.size()[i+2]+pad0)%self.output_patch_size[i]
            padded_length = pad0 + input.size()[i+2] + pad1
            new_shape.append(int(padded_length))
                            
        # Create and fill padded input image.
        indices = [slice(None, None)]*ndim
        spatial_dims = range(2, ndim)
        for dim in spatial_dims:
            pad0 = lost[dim-2]//2
            indices[dim] = slice(pad0, input.size()[dim]+pad0)
        input_padded = Variable(torch.zeros(*new_shape).float())
        if self._use_gpu:
            input_padded = input_padded.cuda()
        input_padded[indices] = input
        
        # Create output image buffer.
        output = Variable(torch.zeros(*tuple(input.size())).float())
        if self._use_gpu:
            output = output.cuda()
        
        # Iterator over indices in an image, for sourcing or placing patches.
        def index_iterator(offset=None):
            if offset is None:
                offset = [0]*(ndim-2)
            def recursive_idx(dim, idx):
                for i in range(0, input.size()[dim]+offset[dim-2],
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
        iter_input = index_iterator(offset=lost//2)
        iter_output = index_iterator()
        for idx_i, idx_o in zip(iter_input, iter_output):
            sl_i = get_patch_slices(idx_i, self.input_patch_size)
            sl_o = get_patch_slices(idx_o, self.output_patch_size)
            input_patch = input_padded[sl_i]
            
            output_patch = self.model(input_patch)
            sl_p = [slice(None, None)]*2
            for dim in range(2, ndim):
                stop = min(output.size()[dim] - sl_o[dim].start,
                           output_patch.size()[dim])
                sl_p.append(slice(0, stop))
            output[sl_o] = output_patch[sl_p]
            
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
