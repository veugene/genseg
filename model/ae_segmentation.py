from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from fcn_maker.loss import dice_loss
from .common.losses import (dist_ratio_mse_abs,
                            mae,
                            mse)


class segmentation_model(nn.Module):
    def __init__(self, encoder, decoder_rec, decoder_seg=None, loss_rec=mae,
                 loss_seg=dice_loss(), lambda_rec=1., lambda_seg=1., rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder        = encoder
        self.decoder_rec    = decoder_rec
        self.decoder_seg    = decoder_seg
        self.loss_rec       = loss_rec
        self.loss_seg       = loss_seg
        self.lambda_rec     = lambda_rec
        self.lambda_seg     = lambda_seg
        self.is_cuda        = False
    
    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        super(segmentation_model, self).cuda(*args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(segmentation_model, self).cpu(*args, **kwargs)
    
    def segment(self, x):
        return self.segmentation(*self.encoder(x))
    
    def evaluate(self, x_A, x_B=None, mask=None, mask_indices=None,
                 optimizer=None, rng=None):
        compute_grad = True if optimizer is not None else False
        if compute_grad:
            if isinstance(optimizer, dict):
                assert len(optimizer)==1
                optimizer = list(optimizer.values())[0]
            optimizer.zero_grad()
        x = x_A
        if x_B is not None:
            x = torch.cat([x_A, x_B], dim=0)
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x, mask, mask_indices, optimizer=optimizer)
    
    def _evaluate(self, x, mask=None, mask_indices=None, optimizer=None):
        loss_rec = 0
        loss_seg = 0
        
        code, skip = self.encoder(x)
        x_rec = None
        y     = None
        
        def _mean(x):
            if not isinstance(x, torch.Tensor) or x.dim()<=1:
                return x
            else:
                return x.view(x.size(0), -1).mean(1)
        
        # Reconstruct.
        if self.lambda_rec:
            x_rec = self.decoder_rec(code)
            loss_rec = _mean(self.loss_rec(x_rec, x))
        
        # Segment.
        if mask is not None and self.lambda_seg:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            y = self.decoder_seg(code, skip)
            loss_seg = _mean(self.loss_seg(y[mask_indices], mask))
        
        # Loss. Compute gradients, if requested.
        loss = ( self.lambda_rec*loss_rec
                +self.lambda_seg*loss_seg)
        if optimizer is not None:
            loss.mean().backward()
            optimizer.step()
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_all',   loss),
            ('l_seg',   loss_seg),
            ('l_rec',   loss_rec),
            ('out_rec', x_rec),
            ('out_seg', y[mask_indices]),
            ('mask',    mask)))
        return outputs
