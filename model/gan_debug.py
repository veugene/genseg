from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from fcn_maker.loss import dice_loss
from .common import (dist_ratio_mse_abs,
                     bce,
                     mae,
                     mse)
from .mine import mine


class gan(nn.Module):
    def __init__(self, g_common, g_residual, g_unique, g_output, disc,
                 z_size=(50,), z_constant=0, rng=None):
        super(gan, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.g_common           = g_common
        self.g_residual         = g_residual
        self.g_unique           = g_unique
        self.g_output           = g_output
        self.disc               = disc
        self.z_size             = z_size
        self.z_constant         = z_constant
        self.is_cuda            = False
        
    def _z_constant(self, batch_size):
        ret = Variable(torch.zeros((batch_size,)+self.z_size,
                                   dtype=torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _z_sample(self, batch_size):
        ret = Variable(torch.randn((batch_size,)
                                    +self.z_size).type(torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        super(gan, self).cuda(*args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(gan, self).cpu(*args, **kwargs)
        
    def decode(self, common, residual, unique):
        out = self.g_output(self.g_common(common),
                            self.g_residual(residual),
                            self.g_unique(unique))
        return out
    
    def evaluate(self, *args, compute_grad=False, **kwargs):
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(*args, compute_grad=compute_grad, **kwargs)
    
    def _evaluate(self, x, grad_penalty=None, disc_clip_norm=None,  
                  compute_grad=False):
        # Generate.
        batch_size = len(x)
        z = {'common'  : self._z_sample(batch_size),
             'residual': self._z_sample(batch_size),
             'unique'  : self._z_sample(batch_size)}
        x_gen = self.decode(**z)
        
        # Generator loss.
        loss_G = bce(self.disc(x_gen), 1)
        
        # Compute generator gradients.
        if compute_grad:
            loss_G.mean().backward()
            self.disc.zero_grad()
        
        # Discriminator losses.
        if grad_penalty and compute_grad:
            x.requires_grad = True
        disc_real = self.disc(x)
        disc_fake = self.disc(x_gen.detach())
        loss_disc_1 = bce(disc_real, 1)
        loss_disc_0 = bce(disc_fake, 0)
        if grad_penalty and compute_grad:
            def _compute_grad_norm(disc_in, disc_out, disc):
                ones = torch.ones_like(disc_out)
                if disc_in.is_cuda:
                    ones = ones.cuda()
                grad = torch.autograd.grad(disc_out,
                                           disc_in,
                                           grad_outputs=ones,
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]
                grad_norm = (grad.view(grad.size(0),-1)**2).sum(1)
                return torch.mean(grad_norm)
            grad_norm = _compute_grad_norm(x, disc_real, self.disc)
            loss_disc_1 += grad_penalty*grad_norm
        loss_D = (loss_disc_0+loss_disc_1)/2.
        if compute_grad:
            loss_D.mean().backward()
            if disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc.parameters(),
                                         max_norm=disc_clip_norm)
        
        # Compile outputs and return.
        outputs = {'l_G'        : loss_G,
                   'l_D'        : loss_D,
                   'out_gen'    : x_gen}
        return outputs
