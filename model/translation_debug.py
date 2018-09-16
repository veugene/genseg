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


class translation_model(nn.Module):
    def __init__(self, encoder_A, decoder_A, encoder_B, decoder_B,
                 disc_A, disc_B, mutual_information, loss_rec=mae,
                 z_size=(50,), lambda_disc=1, lambda_x_id=10, lambda_z_id=1,
                 lambda_const=1, lambda_cyc=0, lambda_mi=1, rng=None,
                 debug_no_constant=False):
        super(translation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder_A          = encoder_A
        self.decoder_A          = decoder_A
        self.encoder_B          = encoder_B
        self.decoder_B          = decoder_B
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.mutual_information = mutual_information
        self.mi_estimator       = mine(mutual_information, rng=self.rng)
        self.loss_rec           = loss_rec
        self.z_size             = z_size
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_const       = lambda_const
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.debug_no_constant  = debug_no_constant
        self.is_cuda            = False
        
    def _z_constant(self, batch_size):
        ret = Variable(torch.zeros((batch_size,)+self.z_size,
                                   dtype=torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        #ret = ret+self._z_constant_param
        return ret
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self.z_size).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _scale(self, z_dict):
        # Unique is constant; scale up the other variations.
        if self.debug_no_constant:
            return z_dict
        lc, lr, lu = len(z_dict['common']), self.z_size[0], self.z_size[0]
        scale = (lc+lr)/float(lc+lr+lu)
        z_scaled = {'common'  : z_dict['common']*scale,
                    'residual': z_dict['residual']*scale,
                    'unique'  : z_dict['unique']}
        return z_scaled
    
    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        super(translation_model, self).cuda(*args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(translation_model, self).cpu(*args, **kwargs)
        
    def encode_A(self, x):
        z_c, z_r, z_u, skip_info = self.encoder_A(x)
        z = {'common'  : z_c,
             'residual': z_r,
             'unique'  : z_u}
        return z, torch.cat([z_c, z_r], dim=1), z_u, skip_info
        
    def decode_A(self, common, residual, unique, skip_info):
        out = self.decoder_A(common, residual, unique, skip_info)
        return out
    
    def encode_B(self, x):
        z_c, z_r, z_u, skip_info = self.encoder_B(x)
        z = {'common'  : z_c,
             'residual': z_r,
             'unique'  : z_u}
        return z, torch.cat([z_c, z_r], dim=1), z_u, skip_info
        
    def decode_B(self, common, residual, unique, skip_info):
        out = self.decoder_B(common, residual, unique, skip_info)
        return out
    
    def translate_AB(self, x_A, rng=None):
        batch_size = len(x_A)
        s_A, _, _, skip_A = self.encode_A(x_A)
        z_AB = {'common'  : s_A['common'],
                'residual': self._z_sample(batch_size, rng=rng),
                'unique'  : self._z_constant(batch_size)}
        if self.debug_no_constant:
            z_AB['unique'] = self._z_sample(batch_size, rng=rng)
        x_AB = self.decode_B(**self._scale(z_AB), skip_info=skip_A)
        return x_AB
    
    def translate_BA(self, x_B, rng=None):
        batch_size = len(x_B)
        s_B, _, _, skip_B = self.encode_B(x_B)
        z_BA = {'common'  : s_B['common'],
                'residual': self._z_sample(batch_size, rng=rng),
                'unique'  : self._z_sample(batch_size, rng=rng)}
        x_BA = self.decode_A(**z_BA, skip_info=skip_B)
        return x_BA
    
    def evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                 compute_grad=False, **kwargs):
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x_A, x_B, mask, mask_indices,
                                  compute_grad=compute_grad, **kwargs)
    
    def _evaluate(self, x_A, x_B, mask=None, mask_indices=None, rng=None,
                  grad_penalty=None, disc_clip_norm=None, compute_grad=False):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, a_A, b_A, skip_A = self.encode_A(x_A)
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id
            or self.lambda_const
            or self.lambda_cyc
            or self.lambda_mi):
                s_B, a_B, b_B, skip_B = self.encode_B(x_B)
        
        # Reconstruct inputs.
        if self.lambda_x_id:
            z_AA = {'common'  : s_A['common'],
                    'residual': s_A['residual'],
                    'unique'  : s_A['unique']}
            z_BB = {'common'  : s_B['common'],
                    'residual': s_B['residual'],
                    'unique'  : self._z_constant(batch_size)}
            if self.debug_no_constant:
                z_BB['unique'] = s_B['unique']
            x_AA = self.decode_A(**z_AA, skip_info=skip_A)
            x_BB = self.decode_B(**self._scale(z_BB), skip_info=skip_B)
        
        # Translate.
        x_AB = x_BA = None
        if self.lambda_disc or self.lambda_z_id:
            z_AB = {'common'  : s_A['common'],
                    'residual': self._z_sample(batch_size, rng=rng),
                    'unique'  : self._z_constant(batch_size)}
            if self.debug_no_constant:
                z_AB['unique'] = self._z_sample(batch_size, rng=rng)
            z_BA = {'common'  : s_B['common'],
                    'residual': self._z_sample(batch_size, rng=rng),
                    'unique'  : self._z_sample(batch_size, rng=rng)}
            x_AB = self.decode_B(**self._scale(z_AB), skip_info=skip_A)
            x_BA = self.decode_A(**z_BA, skip_info=skip_B)
        
        # Reconstruct latent codes.
        if self.lambda_z_id or self.lambda_cyc:
            s_AB, a_AB, b_AB, skip_AB = self.encode_B(x_AB)
            s_BA, a_BA, b_BA, skip_BA = self.encode_A(x_BA)
        
        # Cycle.
        x_ABA = x_BAB = None
        if self.lambda_cyc:
            z_ABA = {'common'  : s_AB['common'],
                     'residual': s_A['residual'],
                     'unique'  : s_A['unique']}
            z_BAB = {'common'  : s_BA['common'],
                     'residual': s_B['residual'],
                     'unique'  : s_B['unique']}
            x_ABA = self.decode_A(**z_ABA, skip_info=skip_AB)
            x_BAB = self.decode_B(**z_BAB, skip_info=skip_BA)
        
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
                                                      z_AB['common'])
            loss_rec['zr_AB'] = self.lambda_z_id*dist(s_AB['residual'],
                                                      z_AB['residual'])
            loss_rec['zu_AB'] = self.lambda_z_id*dist(s_AB['unique'],
                                                      z_AB['unique'])
            loss_rec['zc_BA'] = self.lambda_z_id*dist(s_BA['common'],
                                                      z_BA['common'])
            loss_rec['zr_BA'] = self.lambda_z_id*dist(s_BA['residual'],
                                                      z_BA['residual'])
            loss_rec['zu_BA'] = self.lambda_z_id*dist(s_BA['unique'],
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
        
        # Mutual information loss for generator.
        loss_mi_gen = defaultdict(int)
        if self.lambda_mi:
            loss_mi_gen['A'] =  -( self.lambda_mi
                                  *self.mi_estimator.evaluate(a_A, b_A))
            loss_mi_gen['BA'] = -( self.lambda_mi
                                  *self.mi_estimator.evaluate(a_BA, b_BA))
        
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
                  +_reduce(loss_mi_gen.values())
                  +_reduce([loss_const_B]))
        
        # Compute generator gradients.
        if compute_grad:
            loss_G.mean().backward()
        self.mutual_information.zero_grad()
        if compute_grad and self.lambda_disc:
            self.disc_A.zero_grad()
            self.disc_B.zero_grad()
        
        # Mutual information loss for estimator.
        loss_mi_est = defaultdict(int)
        loss_mi_est['A']  = self.mi_estimator.evaluate(a_A.detach(),
                                                       b_A.detach())
        if compute_grad: loss_mi_est['A'].mean().backward()
        if self.lambda_z_id or self.lambda_cyc:
            loss_mi_est['BA'] = self.mi_estimator.evaluate(a_BA.detach(),
                                                           b_BA.detach())
            if compute_grad: loss_mi_est['BA'].mean().backward()
        
        # Discriminator losses.
        loss_disc = defaultdict(int)
        if self.lambda_disc:
            if grad_penalty and compute_grad:
                x_A.requires_grad = True
                x_B.requires_grad = True
            disc_A_real = self.disc_A(x_A)
            disc_A_fake = self.disc_A(x_BA.detach())
            disc_B_real = self.disc_B(x_B)
            disc_B_fake = self.disc_B(x_AB.detach())
            loss_disc_A_1 = bce(disc_A_real, 1)
            loss_disc_A_0 = bce(disc_A_fake, 0)
            loss_disc_B_1 = bce(disc_B_real, 1)
            loss_disc_B_0 = bce(disc_B_fake, 0)
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
                    grad_norm = (grad.view(grad.size()[0],-1)**2).sum(-1)
                    return torch.mean(grad_norm)
                grad_norm_A = _compute_grad_norm(x_A, disc_A_real, self.disc_A)
                grad_norm_B = _compute_grad_norm(x_B, disc_B_real, self.disc_B)
                loss_disc_A_1 += grad_penalty*grad_norm_A
                loss_disc_B_1 += grad_penalty*grad_norm_B
            loss_disc['A'] = self.lambda_disc*(loss_disc_A_0+loss_disc_A_1)/2.
            loss_disc['B'] = self.lambda_disc*(loss_disc_B_0+loss_disc_B_1)/2.
        loss_D = _reduce([loss_disc['A']+loss_disc['B']])
        if self.lambda_disc and compute_grad:
            loss_D.mean().backward()
            if disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc_A.parameters(),
                                         max_norm=disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc_B.parameters(),
                                         max_norm=disc_clip_norm)
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_G',         loss_G),
            ('l_D',         loss_D),
            ('l_DA',        _reduce([loss_disc['A']])),
            ('l_DB',        _reduce([loss_disc['B']])),
            ('l_gen_AB',    _reduce([loss_gen['AB']])),
            ('l_gen_BA',    _reduce([loss_gen['BA']])),
            ('l_rec_AA',    _reduce([loss_rec['AA']])),
            ('l_rec_BB',    _reduce([loss_rec['BB']])),
            ('l_rec',       _reduce([loss_rec['AA']+loss_rec['BB']])),
            ('l_rec_zc_AB', _reduce([loss_rec['zc_AB']])),
            ('l_rec_zr_AB', _reduce([loss_rec['zr_AB']])),
            ('l_rec_zu_AB', _reduce([loss_rec['zu_AB']])),
            ('l_rec_z_AB',  _reduce(torch.cat([loss_rec['zc_AB'],
                                               loss_rec['zr_AB'],
                                               loss_rec['zu_AB']], dim=1))),
            ('l_rec_zc_BA', _reduce([loss_rec['zc_BA']])),
            ('l_rec_zr_BA', _reduce([loss_rec['zr_BA']])),
            ('l_rec_zu_BA', _reduce([loss_rec['zu_BA']])),
            ('l_rec_z_BA',  _reduce(torch.cat([loss_rec['zc_BA'],
                                               loss_rec['zr_BA'],
                                               loss_rec['zu_BA']], dim=1))),
            ('l_rec_z',     _reduce([ torch.cat([loss_rec['zc_AB'],
                                                 loss_rec['zr_AB'],
                                                 loss_rec['zu_AB']], dim=1)
                                     +torch.cat([loss_rec['zc_BA'],
                                                 loss_rec['zr_BA'],
                                                 loss_rec['zu_BA']], dim=1)])),
            ('l_const_B',   _reduce([loss_const_B])),
            ('l_cyc_ABA',   _reduce([loss_cyc['ABA']])),
            ('l_cyc_BAB',   _reduce([loss_cyc['BAB']])),
            ('l_cyc',       _reduce([loss_cyc['ABA']+loss_cyc['BAB']])),
            ('l_mi_A',      loss_mi_est['A']),
            ('l_mi_BA',     loss_mi_est['BA']),
            ('l_mi',        loss_mi_est['A']+loss_mi_est['BA']),
            ('out_AA',      x_AA),
            ('out_BB',      x_BB),
            ('out_AB',      x_AB),
            ('out_BA',      x_BA),
            ('out_ABA',     x_ABA),
            ('out_BAB',     x_BAB)))
        return outputs
