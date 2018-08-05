import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def dist_ratio_mse_abs(x, y, eps=1e-7):
    return torch.mean((x-y)**2) / (torch.mean(torch.abs(x-y))+eps)
    
    
def mse(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
        target = Variable(target)
    return nn.MSELoss()(prediction, target)
    
    
class segmentation_model(nn.Module):
    def __init__(self, f_factor, f_common, f_residual, f_unique,
                 g_common, g_residual, g_unique, g_output,
                 disc_A, disc_B, mutual_information, loss_segmentation,
                 z_size=(50,), z_constant=0, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_const=1, lambda_cyc=0, lambda_mi=1,
                 lambda_mi=1, lambda_seg=1, disc_clip_norm=1., rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.f_factor           = f_factor
        self.f_common           = f_common
        self.f_residual         = f_residual
        self.f_unique           = f_unique
        self.g_common           = g_common
        self.g_residual         = g_residual
        self.g_unique           = g_unique
        self.g_output           = g_output
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.mutual_information = mutual_information
        self.mi_estimator       = mine(mutual_information, rng=self.rng)
        self.loss_segmentation  = loss_segmentation
        self.z_size             = z_size
        self.z_constant         = z_constant
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_const       = lambda_const
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_seg         = lambda_seg
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
        super(segmentation_model, self).cuda(*args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(segmentation_model, self).cpu(*args, **kwargs)
        
    def encode(self, x):
        z_a, z_b = self.f_factor(x)
        z_common = self.f_common(z_b)
        z_residual = self.f_residual(z_b)
        z_unique = self.f_unique(z_a)
        z = {'common'  : z_common,
             'residual': z_residual,
             'unique'  : z_unique}
        return z, z_a, z_b
        
    def decode(self, common, residual, unique, segment=False):
        out = self.g_output(self.g_common(common),
                            self.g_residual(residual),
                            self.g_unique(unique),
                            segment=segment)
        return out
    
    def translate_AB(self, x_A):
        batch_size = len(x_A)
        s_A, _, _ = self.encode(x_A)
        z_A = {'common'  : s_A['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_constant(batch_size)}
        x_AB = self.decode(**z_A)
        return x_AB
    
    def translate_BA(self, x_B):
        batch_size = len(x_B)
        s_B, _, _ = self.encode(x_B)
        z_B = {'common'  : s_B['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_sample(batch_size)}
        x_BA = self.decode(**z_B)
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_A)
        s_A, _, _ = self.encode(x_A)
        z_AM = {'common'  : self._z_constant(batch_size),
                'residual': self._z_constant(batch_size),
                'unique'  : s_A['unique']}
        x_AM = self.decode(**z_AM, segment=True)
        return x_AM
    
    def evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                 compute_grad=False):
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x_A, x_B, mask, mask_indices, compute_grad)
    
    def _evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                  compute_grad=False):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, a_A, b_A = self.encode(x_A)
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id
            or self.lambda_const
            or self.lambda_cyc
            or self.lambda_mi):
                s_B, a_B, b_B = self.encode(x_B)
        
        # Reconstruct inputs.
        if self.lambda_x_id:
            z_AA = {'common'  : s_A['common'],
                    'residual': s_A['residual'],
                    'unique'  : s_A['unique']}
            z_BB = {'common'  : s_B['common'],
                    'residual': s_B['residual'],
                    'unique'  : self._z_constant(batch_size)}
            x_AA = self.decode(**z_AA)
            x_BB = self.decode(**z_BB)
        
        # Translate.
        x_AB = x_BA = None
        if self.lambda_disc or self.lambda_z_id:
            z_AB = {'common'  : s_A['common'],
                    'residual': self._z_sample(batch_size),
                    'unique'  : self._z_constant(batch_size)}
            z_BA = {'common'  : s_B['common'],
                    'residual': self._z_sample(batch_size),
                    'unique'  : self._z_sample(batch_size)}
            x_AB = self.decode(**z_AB)
            x_BA = self.decode(**z_BA)
        
        # Reconstruct latent codes.
        if self.lambda_z_id:
            s_AB, a_AB, b_AB = self.encode(x_AB)
            s_BA, a_BA, b_BA = self.encode(x_BA)
        
        # Cycle.
        x_ABA = x_BAB = None
        if self.lambda_cyc:
            z_ABA = {'common'  : s_AB['common'],
                     'residual': s_A['residual'],
                     'unique'  : s_A['unique']}
            z_BAB = {'common'  : s_BA['common'],
                     'residual': s_B['residual'],
                     'unique'  : s_B['unique']}
            x_ABA = self.decode(**z_ABA)
            x_BAB = self.decode(**z_BAB)
            
        # Segment.
        x_AM = None
        if self.lambda_seg and mask is not None:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            z_AM = {'common'  : self._z_constant(num_masks),
                    'residual': self._z_constant(num_masks),
                    'unique'  : s_A['unique'][mask_indices]}
            x_AM = self.decode(**z_AM)
            if self.lambda_z_id:
                s_AM, a_AM, b_AM = self.encode(x_AM)
        
        # Generator losses.
        loss_G = 0
        def dist(x, y): return torch.mean(torch.abs(x-y))
        if self.lambda_disc:
            loss_G += self.lambda_disc * mse(self.disc_B(x_AB), 1)
            loss_G += self.lambda_disc * mse(self.disc_A(x_BA), 1)
        if self.lambda_x_id:
            loss_G += self.lambda_x_id * dist(x_AA, x_A)
            loss_G += self.lambda_x_id * dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_G += self.lambda_z_id * dist(s_AB['common'],
                                              z_AB['common'].detach())
            loss_G += self.lambda_z_id * dist(s_AB['residual'],
                                              z_AB['residual'])   # detached
            loss_G += self.lambda_z_id * dist(s_AB['unique'],
                                              z_AB['unique'])     # detached
            loss_G += self.lambda_z_id * dist(s_BA['common'],
                                              z_BA['common'].detach())
            loss_G += self.lambda_z_id * dist(s_BA['residual'],
                                              z_BA['residual'])   # detached
            loss_G += self.lambda_z_id * dist(s_BA['unique'],
                                              z_BA['unique'])     # detached
        if self.lambda_const:
            loss_G += self.lambda_const * dist(s_B['unique'],
                                               self._z_constant(batch_size))
        if self.lambda_cyc:
            loss_G += self.lambda_cyc * dist(x_ABA, x_A)
            loss_G += self.lambda_cyc * dist(x_BAB, x_B)
        if self.lambda_mi:
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_A, b_A)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_B, b_B)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_AB, b_AB)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_BA, b_BA)
        
        # Segment.
        loss_segmentation = 0
        if self.lambda_seg and mask is not None:
            loss_segmentation = self.loss_segmentation(x_AM,
                                                       mask[mask_indices])
            loss_G += self.lambda_seg * loss_segmentation
            if self.lambda_z_id:
                loss_G += (  self.lambda_z_id
                           * dist(s_AM['unique'], z_AM['unique'].detach()))
        
        # Compute generator gradients.
        if compute_grad:
            loss_G.backward()
        if compute_grad and self.lambda_disc:
            self.disc_A.zero_grad()
            self.disc_B.zero_grad()
        
        # Discriminator losses.
        loss_disc_A = loss_disc_B = 0
        if self.lambda_disc:
            loss_disc_A = (  mse(self.disc_A(x_A), 1)
                           + mse(self.disc_A(x_BA.detach()), 0))
            loss_disc_B = (  mse(self.disc_B(x_B), 1)
                           + mse(self.disc_B(x_AB.detach()), 0))
            loss_G += self.lambda_disc * (loss_disc_A+loss_disc_B)
        if self.lambda_disc and compute_grad:
            loss_disc_A.backward()
            loss_disc_B.backward()
            if self.disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc_A.parameters(),
                                         max_norm=self.disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc_B.parameters(),
                                         max_norm=self.disc_clip_norm)
        
        # Compile outputs and return.
        losses  = {'seg'   : loss_segmentation,
                   'disc_A': loss_disc_A,
                   'disc_B': loss_disc_B,
                   'loss_G': loss_G}
        outputs = {'x_AB'  : x_AB,
                   'x_BA'  : x_BA,
                   'x_ABA' : x_ABA,
                   'x_BAB' : x_BAB,
                   'x_AM'  : x_AM}
        return losses, outputs
