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


def _reduce(loss):
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    return sum([_mean(v) for v in loss])


def _cat(x, dim):
    _x = [t for t in x if isinstance(t, torch.Tensor)]
    if len(_x):
        return torch.cat(_x, dim=dim)
    return 0


class segmentation_model(nn.Module):
    def __init__(self, encoder, decoder, disc_A, disc_B, shape_sample,
                 loss_rec=mae, loss_seg=None, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_seg=1, lambda_cross=0, lambda_cyc=0,
                 rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder          = encoder
        self.decoder          = decoder
        self.disc = {'A'    :     disc_A,
                     'B'    :     disc_B}   # separate params
        self.loss_rec           = loss_rec
        self.loss_seg           = loss_seg if loss_seg else dice_loss()
        self.shape_sample       = shape_sample
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_cross       = lambda_cross
        self.lambda_cyc         = lambda_cyc
        self.is_cuda            = False
    
    def _z_constant(self, batch_size):
        ret = Variable(torch.zeros((batch_size,)+self.shape_sample,
                                    dtype=torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self.shape_sample).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _call_on_all_models(self, fname, *args, **kwargs):
        self.disc['A'] = getattr(self.disc['A'], fname)(*args, **kwargs)
        self.disc['B'] = getattr(self.disc['B'], fname)(*args, **kwargs)
        getattr(super(segmentation_model, self), fname)(*args, **kwargs)
    
    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        self._call_on_all_models('cuda', *args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        self._call_on_all_models('cpu', *args, **kwargs)
    
    def train(self, mode=True):
        self._call_on_all_models('train', mode=mode)
    
    def eval(self):
        self._call_on_all_models('eval')
        
    def translate_AB(self, x_A, rng=None):
        s_A, skip_A = self.encoder(x_A)
        x_AB = x_A - self.decoder(s_A, skip_info=skip_A)   # minus
        return x_AB
    
    def translate_BA(self, x_B, rng=None):
        batch_size = len(x_B)
        s_B, skip_B = self.encoder(x_B)
        x_BA = self.decoder(self._z_sample(batch_size, rng=rng),
                            skip_info=skip_B)
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_A)
        s_A, skip_A = self.encoder(x_A)
        x_AM = self.decoder(s_A, skip_info=skip_A, out_idx=1)
        return x_AM
    
    def evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                 optimizer=None, **kwargs):
        compute_grad = True if optimizer is not None else False
        if compute_grad:
            for key in optimizer:
                optimizer[key].zero_grad()
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x_A, x_B, mask, mask_indices,
                                  optimizer=optimizer, **kwargs)
    
    def _evaluate(self, x_A, x_B, mask=None, mask_indices=None, rng=None,
                  grad_penalty=None, disc_clip_norm=None, optimizer=None):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, skip_A = self.encoder(x_A)
        only_seg = True
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id
            or self.lambda_mi):
                s_B, skip_B = self.encoder(x_B)
                only_seg = False
        
        # Reconstruct input.
        x_BB = None
        if self.lambda_x_id:
            x_BB = x_B - self.decoder(s_B, skip_info=skip_B)    # minus
        
        # Translate.
        x_AB = x_BA = x_AB_residual = x_BA_residual = None
        if self.lambda_disc or self.lambda_z_id:
            z_BA = self._z_sample(batch_size, rng=rng)
            x_BA_residual = self.decoder(z_BA, skip_info=skip_B)
            x_AB_residual = self.decoder(s_A,  skip_info=skip_A)
            x_BA = x_B + x_BA_residual      # plus
            x_AB = x_A - x_AB_residual      # minus
        x_cross = x_cross_residual = None
        if self.lambda_disc and self.lambda_cross:
            x_cross_residual = self.decoder(s_A, skip_info=skip_B)
            x_cross = x_B + x_cross_residual  # plus
        
        # Segment.
        x_AM = None
        if self.lambda_seg and mask is not None and len(mask)>0:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            skip_A_filtered = [s[mask_indices] for s in skip_A]
            x_AM = self.decoder(s_A[mask_indices],
                                skip_info=skip_A_filtered,
                                out_idx=1)
        
        # Reconstruct latent code.
        if self.lambda_z_id:
            s_BA, skip_BA = self.encoder(x_BA)
            if self.lambda_cross:
                s_cross, skip_cross = self.encoder(x_cross)
        
        # Cycle.
        x_cross_A = x_cross_A_residual = None
        if self.lambda_cyc:
            s_AB, skip_AB = self.encoder(x_AB)
            x_cross_A_residual = self.decoder(s_cross, skip_info=skip_AB)
            x_cross_A = x_AB + x_cross_A_residual   # plus
        
        # Discriminator losses.
        def mse_(prediction, target):
            if not isinstance(prediction, torch.Tensor):
                return sum([mse_(elem, target) for elem in prediction])
            return _reduce([mse(prediction, target)])
        loss_disc = defaultdict(int)
        if self.lambda_disc:
            if grad_penalty and optimizer is not None:
                x_A.requires_grad = True
                x_B.requires_grad = True
            disc_A_real = self.disc['A'](x_A)
            disc_A_fake = self.disc['A'](x_BA.detach())
            disc_B_real = self.disc['B'](x_B)
            disc_B_fake = self.disc['B'](x_AB.detach())
            loss_disc_A_1 = mse_(disc_A_real, 1)
            loss_disc_A_0 = mse_(disc_A_fake, 0)
            loss_disc_B_1 = mse_(disc_B_real, 1)
            loss_disc_B_0 = mse_(disc_B_fake, 0)
            if self.lambda_cross:
                loss_disc_cross = mse_(self.disc['A'](x_cross.detach()), 0)
                loss_disc_A_0 = ( loss_disc_A_0
                                 +loss_disc_cross*self.lambda_cross)
            if grad_penalty and optimizer is not None:
                def _compute_grad_norm(disc_in, disc_out, disc):
                    ones = torch.ones_like(disc_out)
                    if disc_in.is_cuda:
                        ones = ones.cuda()
                    grad = torch.autograd.grad(disc_out.sum(),
                                               disc_in,
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]
                    grad_norm = (grad.view(grad.size()[0],-1)**2).sum(-1)
                    return torch.mean(grad_norm)
                grad_norm_A = _compute_grad_norm(x_A, disc_A_real,
                                                 self.disc['A'])
                grad_norm_B = _compute_grad_norm(x_B, disc_B_real,
                                                 self.disc['B'])
                loss_disc_A_1 += grad_penalty*grad_norm_A
                loss_disc_B_1 += grad_penalty*grad_norm_B
            loss_disc['A'] = self.lambda_disc*(loss_disc_A_0+loss_disc_A_1)
            loss_disc['B'] = self.lambda_disc*(loss_disc_B_0+loss_disc_B_1)
        loss_D = _reduce([loss_disc['A']+loss_disc['B']])
        if self.lambda_disc and optimizer is not None:
            loss_D.mean().backward()
            if disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc['A'].parameters(),
                                         max_norm=disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc['B'].parameters(),
                                         max_norm=disc_clip_norm)
            optimizer['D'].step()
        
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            loss_gen['AB'] = self.lambda_disc*mse_(self.disc['B'](x_AB), 1)
            loss_gen['BA'] = self.lambda_disc*mse_(self.disc['A'](x_BA), 1)
        if self.lambda_disc and self.lambda_cross:
            loss_gen['cross'] = ( self.lambda_disc
                                 *self.lambda_cross
                                 *mse_(self.disc['A'](x_cross), 1))
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        dist = self.loss_rec
        if self.lambda_x_id:
            loss_rec['BB'] = self.lambda_x_id*dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_rec['z_BA']    = self.lambda_z_id*dist(s_BA, z_BA)
            if self.lambda_cross:
                loss_rec['z_cross'] = self.lambda_z_id*dist(s_cross, s_A)
        
        # Cross cycle consistency loss.
        loss_cyc = 0
        if self.lambda_cyc:
            loss_cyc = self.lambda_cyc*dist(x_cross_A, x_A)
        
        # Segmentation loss.
        loss_seg = 0
        if self.lambda_seg and mask is not None and len(mask)>0:
            loss_seg = self.lambda_seg*self.loss_seg(x_AM, mask)
        
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values())
                  +_reduce([loss_seg]))
        
        # Compute generator gradients.
        if optimizer is not None and isinstance(loss_G, torch.Tensor):
            loss_G.mean().backward()
            optimizer['G'].step()
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_G',             loss_G),
            ('l_D',             loss_D),
            ('l_DA',            _reduce([loss_disc['A']])),
            ('l_DB',            _reduce([loss_disc['B']])),
            ('l_gen_AB',        _reduce([loss_gen['AB']])),
            ('l_gen_BA',        _reduce([loss_gen['BA']])),
            ('l_gen_cross',     _reduce([loss_gen['cross']])),
            ('l_rec',           _reduce([loss_rec['BB']])),
            ('l_rec_z',         _reduce([_cat([loss_rec['z_BA'],
                                               loss_rec['z_cross']], dim=1)])),
            ('l_rec_z_BA',      _reduce([loss_rec['z_BA']])),
            ('l_rec_z_cross',   _reduce([loss_rec['z_cross']])),
            ('l_cyc',           _reduce([loss_cyc])),
            ('l_seg',           _reduce([loss_seg])),
            ('out_seg',         x_AM),
            ('out_BB',          x_BB),
            ('out_AB',          x_AB),
            ('out_AB_res',      x_AB_residual),
            ('out_BA',          x_BA),
            ('out_BA_res',      x_BA_residual),
            ('out_cross',       x_cross),
            ('out_cross_res',   x_cross_residual),
            ('out_cross_A',     x_cross_A),
            ('out_cross_A_res', x_cross_A_residual),
            ('mask',          mask)))
        return outputs
