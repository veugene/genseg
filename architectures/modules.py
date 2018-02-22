from __future__ import (print_function,
                        division)
import numpy as np
import torch
from torch.autograd import Variable


def _to_cuda(x, device=None):
    x.data = x.data.cuda(device)
    if x._grad is not None:
        x._grad.data = x._grad.data.cuda(device)
    return x


class _overlap_tile_function(torch.autograd.Function):
    @staticmethod
    def _patch_slice_iterator(input_patch_size, output_patch_size,
                              input_size):
        lost = np.subtract(input_patch_size, output_patch_size)
        ndim = len(input_patch_size)+2
        
        #Iterator over indices in an image, for sourcing or placing patches.
        def index_iterator(offset=None):
            if offset is None:
                offset = [0]*(ndim-2)
            def recursive_idx(dim, idx):
                for i in range(0, input_size[dim]+offset[dim-2],
                               output_patch_size[dim-2]):
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
            sl_i = get_patch_slices(idx_i, input_patch_size)
            sl_o = get_patch_slices(idx_o, output_patch_size)
            yield sl_i, sl_o
    
    @staticmethod
    def _create_padded_buffer(input_patch_size, output_patch_size, input_size,
                              use_gpu=False, device=None):
        ndim = len(input_patch_size)+2
        
        # Compute the required size of the padded input image.
        lost = np.subtract(input_patch_size, output_patch_size)
        new_shape = list(input_size[:2])
        for i in range(ndim-2):
            pad0 = lost[i]//2
            pad1 = lost[i]-pad0
            padded_length = pad0 + input_size[i+2] + pad1
            new_shape.append(int(padded_length))
                            
        # Create padded buffer.
        input_padded = Variable(torch.zeros(*new_shape).float())
        if use_gpu:
            input_padded = _to_cuda(input_padded, device)
            
        # Determine indices of the region bounded by padding.
        indices = [slice(None, None)]*ndim
        spatial_dims = range(2, ndim)
        for dim in spatial_dims:
            pad0 = lost[dim-2]//2
            indices[dim] = slice(pad0, input_size[dim]+pad0)
        
        return input_padded, indices
    
    @staticmethod
    def forward(ctx, input, model, input_patch_size, output_patch_size,
                out_channels, *args):
        ndim = len(input_patch_size)+2
        input_size = input.size()
        use_gpu = True if input.is_cuda else False
        device = None if not input.is_cuda else input.get_device()
        
        # Create and fill padded input image.
        input_padded, indices = _overlap_tile_function._create_padded_buffer(\
                                                        input_patch_size,
                                                        output_patch_size,
                                                        input_size=input_size,
                                                        use_gpu=use_gpu,
                                                        device=device)
        
        input = Variable(input.data)
        if use_gpu:
            input = _to_cuda(input, device)
        input_padded[indices] = input
        
        # Create output image buffer.
        output_size = list(input_size)
        output_size[1] = out_channels
        output = Variable(torch.zeros(*tuple(output_size)).float(),
                          requires_grad=True)
        if use_gpu:
            output = _to_cuda(output, device)
        
        # Process input patch-wise.
        input_patch_list = []
        output_patch_list = []
        iter_type = _overlap_tile_function._patch_slice_iterator
        for sl_i, sl_o in iter_type(input_patch_size, output_patch_size,
                                    input_size):
            input_patch = Variable(input_padded[sl_i].data,
                                   requires_grad=True)
            if use_gpu:
                input_patch = _to_cuda(input_patch, device)
            with torch.enable_grad():
                output_patch = model(input_patch)
            output[sl_o] = output_patch
            output_patch_list.append(output_patch)
            input_patch_list.append(input_patch)
            
        # Save input, activations, params for backprop.
        ctx.input_patch_list = input_patch_list
        ctx.output_patch_list = output_patch_list
        ctx.save_for_backward(*args)
        
        # Store values for backprop.
        ctx.model = model
        ctx.input_patch_size = input_patch_size
        ctx.output_patch_size = output_patch_size
        ctx.use_gpu = use_gpu
        ctx.device = device
        ctx.input_size = input_size

        return output
    
    @staticmethod
    def backward(ctx, output_grad):
        # Create buffer for input gradients.
        input_grad_padded, indices = \
                                _overlap_tile_function._create_padded_buffer(\
                                                    ctx.input_patch_size,
                                                    ctx.output_patch_size,
                                                    input_size=ctx.input_size,
                                                    use_gpu=ctx.use_gpu,
                                                    device=ctx.device)   
        
        # Get parameters.
        params = ctx.saved_variables
            
        # Process input patch-wise.
        params_grad = [None]*len(params)
        iter_type = _overlap_tile_function._patch_slice_iterator
        for i, (sl_i, sl_o) in enumerate(iter_type(ctx.input_patch_size,
                                                   ctx.output_patch_size,
                                                   ctx.input_size)):
            input_patch = ctx.input_patch_list[i]
            output_patch = ctx.output_patch_list[i]
            output_grad_patch = output_grad[sl_o]

            # Get gradients
            grad_patch = torch.autograd.grad(output_patch,
                                             (input_patch,)+tuple(params),
                                             output_grad_patch,
                                             retain_graph=True)
            input_grad_patch = grad_patch[0]
            params_grad_patch = grad_patch[1:]
            
            # Accumulate gradients.
            input_grad_padded[sl_i] += input_grad_patch
            for i, p_grad in enumerate(params_grad_patch):
                if params_grad[i] is None:
                    params_grad[i] = p_grad
                else:
                    params_grad[i] += p_grad
                        
            # Trim padded input grad to remove padding.
            input_grad = input_grad_padded[indices]
        
        return (input_grad,) + (None,)*4 + tuple(params_grad)


class overlap_tile(torch.nn.Module):
    """
    Implements the overlap-tile strategy to save memory at the cost of extra
    compute time (both training and inference).
    
    Args:
        input_patch_size (tuple of int) : The spatial size of the overlapping
            tiles to take from the input. The output tile size is determined
            automatically.
        model (Module) : A pytorch module implementing the model/layer to
            do overlap-tile with. Convolutions must have no zero-padding,
            must have stride of 1, and no spatial pooling; the output after
            overlap-tile is assumed to have the same spatial size as the input.
        in_channels (int) : The number of input channels.
        out_channels (int) : The number of output channels.
        verbose (bool) : Whether to print the autodetermined output patch
            size.
    """
    def __init__(self, input_patch_size, model, in_channels, out_channels,
                 verbose=True):
        super(overlap_tile, self).__init__()
        self.input_patch_size = input_patch_size
        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_patch_size = None
        self.verbose = verbose
        
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self, memo=None, prefix=''):
        return self.model.named_parameters()
        
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
            input_size = (1,self.in_channels)+tuple(self.input_patch_size)
            _in = Variable(torch.zeros(*input_size).float())
            if input.is_cuda:
                _in = _to_cuda(_in, input.get_device())
            _out = self.model(_in)
            self.output_patch_size = tuple(_out.size()[2:])
            if self.verbose:
                print("OVERLAP-TILE: For input patch size {}, output patch "
                      "size is {}."
                      "".format(self.input_patch_size, self.output_patch_size))
            
        # Forward pass.
        output = _overlap_tile_function.apply(input,
                                              self.model,
                                              self.input_patch_size,
                                              self.output_patch_size,
                                              self.out_channels,
                                              *self.model.parameters())
            
        return output
 

if __name__=='__main__':
    class identity_module(torch.nn.Module):
        def forward(self, input):
            return input

    model = identity_module()
    #model = torch.nn.Conv2d(1, 1, 3, padding=0)
    tiler = overlap_tile(input_patch_size=(3,5), model=model,
                         in_channels=1, out_channels=1)

    edge = 10
    im_input = np.reshape(range(edge**2), (1,1,edge,edge)).astype(np.float32)
    im_input = Variable(torch.from_numpy(im_input))
    im_output = tiler(im_input)

    if np.all(im_output==(im_input)):
        print("Test PASSED")
    else:
        print("Test FAILED")
    #print(im_input.size(), im_output.size())
        
    #loss = im_output.mean()-1
    #loss.backward()
