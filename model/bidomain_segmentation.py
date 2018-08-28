from collections import (defaultdict,
                         OrderedDict)
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


class segmentation_model(nn.Module):
    def __init__(self, f_factor, f_common, f_residual, f_unique, g_common,
                 g_residual, g_unique, g_output, disc_A, disc_B,
                 mutual_information, loss_seg=dice_loss(), loss_rec=mae,
                 z_size=(50,), z_constant=0, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_const=1, lambda_cyc=0, lambda_mi=1,
                 lambda_seg=1, grad_penalty=None, disc_clip_norm=None,
                 rng=None):
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
        self.loss_seg           = loss_seg
        self.loss_rec           = loss_rec
        self.z_size             = z_size
        self.z_constant         = z_constant
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_const       = lambda_const
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_seg         = lambda_seg
        self.grad_penalty       = grad_penalty
        self.disc_clip_norm     = disc_clip_norm
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
        z_a, z_b   = self.f_factor(x)
        z_common   = self.f_common(z_b)
        z_residual = self.f_residual(z_b)
        z_unique   = self.f_unique(z_a)
        z = {'common'  : z_common,
             'residual': z_residual,
             'unique'  : z_unique}
        return z, z_a, z_b
        
    def decode(self, common, residual, unique):
        out = self.g_output(self.g_common(common),
                            self.g_residual(residual),
                            self.g_unique(unique))
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
        x_AM = self.decode(**z_AM)
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
        if self.lambda_z_id or self.lambda_cyc:
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
        if self.lambda_seg and mask is not None and len(mask)>0:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            z_AM = {'common'  : self._z_constant(num_masks),
                    'residual': self._z_constant(num_masks),
                    'unique'  : s_A['unique'][mask_indices]}
            x_AM = self.decode(**z_AM)
        
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            loss_gen['AB'] = self.lambda_disc*bce(self.disc_B(x_AB), 1)
            loss_gen['BA'] = self.lambda_disc*bce(self.disc_A(x_BA), 1)
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        dist = self.loss_rec
        if self.lambda_x_id:
            loss_rec['AA'] = self.lambda_x_id*dist(x_AA, x_A)
            loss_rec['BB'] = self.lambda_x_id*dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_rec['zc_AB'] = self.lambda_z_id*dist(s_AB['common'],
                                                      z_AB['common'].detach())
            loss_rec['zr_AB'] = self.lambda_z_id*dist(s_AB['residual'],
                                                      z_AB['residual'])
            loss_rec['zu_AB'] = self.lambda_z_id*dist(s_AB['unique'],
                                                      z_AB['unique'])
            loss_rec['zc_BU'] = self.lambda_z_id*dist(s_BA['common'],
                                                      z_BA['common'].detach())
            loss_rec['zr_BU'] = self.lambda_z_id*dist(s_BA['residual'],
                                                      z_BA['residual'])
            loss_rec['zu_BU'] = self.lambda_z_id*dist(s_BA['unique'],
                                                      z_BA['unique'])
        
        # Constant 'unique' representation for B -- loss.
        loss_const_B = 0
        if self.lambda_const:
            loss_const_B = self.lambda_const*dist(s_B['unique'],
                                                  self._z_constant(batch_size))
        
        # Cycle consistency loss.
        loss_cyc = defaultdict(int)
        if self.lambda_cyc:
            loss_cyc['ABA'] = self.lambda_cyc*dist(x_ABA, x_A)
            loss_cyc['BAB'] = self.lambda_cyc*dist(x_BAB, x_B)
        
        # Mutual mutual information loss.
        loss_mi = defaultdict(int)
        if self.lambda_mi:
            loss_mi['A'] = ( self.lambda_mi
                            *self.mi_estimator.evaluate(a_A, b_A))
            loss_mi['BA'] = ( self.lambda_mi
                             *self.mi_estimator.evaluate(a_BA, b_BA))
        
        # Segmentation loss.
        loss_seg = 0
        if self.lambda_seg and mask is not None and len(mask)>0:
            loss_seg = self.lambda_seg*self.loss_seg(x_AM, mask)
        
        # All generator losses combined.
        def _reduce(loss):
            def _mean(x):
                if not isinstance(x, torch.Tensor) or x.dim()<=1:
                    return x
                else:
                    return x.view(x.size(0), -1).mean(1)
            if not hasattr(loss, '__len__'): loss = [loss]
            return sum([_mean(v) for v in loss])
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values())
                  +_reduce(loss_cyc.values())
                  +_reduce(loss_mi.values())
                  +_reduce([loss_const_B])
                  +_reduce([loss_seg]))
        
        # Compute generator gradients.
        if compute_grad:
            loss_G.mean().backward()
        if compute_grad and self.lambda_disc:
            self.disc_A.zero_grad()
            self.disc_B.zero_grad()
        
        # Discriminator losses.
        loss_disc = defaultdict(int)
        if self.lambda_disc:
            if self.grad_penalty and compute_grad:
                x_A.requires_grad = True
                x_B.requires_grad = True
            disc_A_real = self.disc_A(x_A)
            disc_A_fake = self.disc_A(x_BA.detach())
            disc_B_real = self.disc_A(x_B)
            disc_B_fake = self.disc_A(x_AB.detach())
            loss_disc_A_1 = bce(disc_A_real, 1)
            loss_disc_A_0 = bce(disc_A_fake, 0)
            loss_disc_B_1 = bce(disc_B_real, 1)
            loss_disc_B_0 = bce(disc_B_fake, 0)
            if self.grad_penalty and compute_grad:
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
                    grad_norm = (grad.view(grad.size()[0],-1)**2).sum(-1)
                    return torch.mean(grad_norm)
                grad_norm_A = _compute_grad_norm(x_A, disc_A_real, self.disc_A)
                grad_norm_B = _compute_grad_norm(x_B, disc_B_real, self.disc_B)
                loss_disc_A_1 += self.grad_penalty*grad_norm_A
                loss_disc_B_1 += self.grad_penalty*grad_norm_B
            loss_disc['A'] = self.lambda_disc*(loss_disc_A_0+loss_disc_A_1)/2.
            loss_disc['B'] = self.lambda_disc*(loss_disc_B_0+loss_disc_B_1)/2.
        loss_D = _reduce(loss_disc['A']+loss_disc['B'])
        if self.lambda_disc and compute_grad:
            loss_D.mean().backward()
            if self.disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc_A.parameters(),
                                         max_norm=self.disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc_B.parameters(),
                                         max_norm=self.disc_clip_norm)
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_G',         loss_G),
            ('l_D',         loss_D),
            ('l_gen_AB',    loss_gen['AB']),
            ('l_gen_BA',    loss_gen['BA']),
            ('l_rec_AA',    loss_rec['AA']),
            ('l_rec_BB',    loss_rec['BB']),
            ('l_rec',       loss_rec['AA']+loss_rec['BB']),
            ('l_rec_zc_AB', loss_rec['zc_AB']),
            ('l_rec_zr_AB', loss_rec['zr_AB']),
            ('l_rec_zu_AB', loss_rec['zu_AB']),
            ('l_rec_z_AB',  loss_rec['zc_AB']+loss_rec['zr_AB']
                           +loss_rec['zu_AB']),
            ('l_rec_zc_BA', loss_rec['zc_BA']),
            ('l_rec_zr_BA', loss_rec['zr_BA']),
            ('l_rec_zu_BA', loss_rec['zu_BA']),
            ('l_rec_z_BA',  loss_rec['zc_BA']+loss_rec['zr_BA']
                           +loss_rec['zu_BA']),
            ('l_rec_z',     loss_rec['zc_AB']+loss_rec['zc_BA']
                           +loss_rec['zr_AB']+loss_rec['zr_BA']
                           +loss_rec['zu_AB']+loss_rec['zu_BA']),
            ('l_const_B',   loss_const_B),
            ('l_cyc_ABA',   loss_cyc['ABA']),
            ('l_cyc_BAB',   loss_cyc['BAB']),
            ('l_cyc',       loss_cyc['ABA']+loss_cyc['BAB']),
            ('l_mi_A',      loss_mi['A']),
            ('l_mi_AB',     loss_mi['AB']),
            ('l_mi',        loss_mi['A']+loss_mi['AB']),
            ('l_seg',       loss_seg),
            ('out_AB',      x_AB),
            ('out_BA',      x_BA),
            ('out_ABA',     x_ABA),
            ('out_BAB',     x_BAB),
            ('out_seg',     x_AM),
            ('mask',        mask)))
        return outputs
