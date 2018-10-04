from collections import (defaultdict,
                         OrderedDict)
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from fcn_maker.loss import dice_loss
from .common.network.basic import grad_norm
from .common.losses import (bce,
                            dist_ratio_mse_abs,
                            gan_objective,
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
                 disc_cross=None, loss_rec=mae, loss_seg=None,
                 loss_gan='hinge', relativistic=False, grad_penalty=None,
                 disc_clip_norm=None, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_seg=1, lambda_cross=0, lambda_cyc=0,
                 lambda_sample=1, sample_image_space=False,
                 sample_decoder=None, rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder          = encoder
        self.decoder          = decoder
        if disc_cross is None:
            disc_cross = disc_A
        self.disc = {'A'    :     disc_A,
                     'B'    :     disc_B,
                     'C'    :     disc_cross}   # separate params
        self.shape_sample       = shape_sample
        self.loss_rec           = loss_rec
        self.loss_seg           = loss_seg if loss_seg else dice_loss()
        self.loss_gan           = loss_gan
        self.relativistic       = relativistic
        self.grad_penalty       = grad_penalty
        self.disc_clip_norm     = disc_clip_norm
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_cross       = lambda_cross
        self.lambda_cyc         = lambda_cyc
        self.lambda_sample      = lambda_sample
        self.sample_image_space = sample_image_space
        self.sample_decoder     = sample_decoder
        self.is_cuda            = False
        self._gan               = gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)
    
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
        for key in self.disc:
            self.disc[key] = getattr(self.disc[key], fname)(*args, **kwargs)
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
                  optimizer=None):
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
        
        # Translate.
        x_AB = x_BA = x_AB_residual = x_BA_residual = None
        if self.lambda_disc or self.lambda_z_id:
            z_BA_im = self._z_sample(batch_size, rng=rng)
            if self.sample_image_space:
                z_BA_im = self._z_sample(batch_size, rng=rng)
                z_BA, _ = self.encoder(z_BA_im)
            else:
                z_BA    = z_BA_im
            x_BA_residual = self.decoder(z_BA, skip_info=skip_B)
            x_AB_residual = self.decoder(s_A,  skip_info=skip_A)
            x_BA = x_B + x_BA_residual      # plus
            x_AB = x_A - x_AB_residual      # minus
        x_cross = x_cross_residual = None
        if self.lambda_disc and self.lambda_cross:
            x_cross_residual = self.decoder(s_A, skip_info=skip_B)
            x_cross = x_B + x_cross_residual  # plus
                
        # Reconstruct input.
        x_BB = z_BA_im_rec = None
        if self.lambda_x_id:
            x_BB = x_B - self.decoder(s_B, skip_info=skip_B)    # minus
        if self.sample_image_space and self.sample_decoder is not None:
            z_BA_im_rec = self.sample_decoder(z_BA)
        
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
        loss_disc = defaultdict(int)
        gradnorm_D = 0
        if self.lambda_disc:
            loss_disc_A = self._gan.D(self.disc['A'],
                                      fake=x_BA.detach(), real=x_A)
            loss_disc_B = self._gan.D(self.disc['B'],
                                      fake=x_AB.detach(), real=x_B)
            loss_disc['A'] = self.lambda_disc*loss_disc_A
            loss_disc['B'] = self.lambda_disc*loss_disc_B
            if self.lambda_cross:
                loss_disc_C = self._gan.D(self.disc['C'],
                                          fake=x_cross.detach(), real=x_A)
                loss_disc['C'] = ( self.lambda_disc*self.lambda_cross
                                  *loss_disc_C)
        loss_D = _reduce([loss_disc['A']+loss_disc['B']+loss_disc['C']])
        if self.lambda_disc and optimizer is not None:
            loss_D.mean().backward()
            if self.disc_clip_norm:
                nn.utils.clip_grad_norm_(self.disc['A'].parameters(),
                                         max_norm=self.disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc['B'].parameters(),
                                         max_norm=self.disc_clip_norm)
                nn.utils.clip_grad_norm_(self.disc['C'].parameters(),
                                         max_norm=self.disc_clip_norm)
            optimizer['D'].step()
            gradnorm_D = sum([grad_norm(self.disc[k])
                              for k in self.disc.keys()
                              if self.disc[k] is not None])
        
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            loss_gen['AB'] = self.lambda_disc*self._gan.G(self.disc['B'],
                                                          fake=x_AB, real=x_B)
            loss_gen['BA'] = self.lambda_disc*self._gan.G(self.disc['A'],
                                                          fake=x_BA, real=x_A)
        if self.lambda_disc and self.lambda_cross:
            loss_gen['C'] = ( self.lambda_disc*self.lambda_cross
                             *self._gan.G(self.disc['C'],
                                          fake=x_cross, real=x_A))
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        dist = self.loss_rec
        if self.lambda_x_id:
            loss_rec['BB'] = self.lambda_x_id*dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_rec['z_BA'] = self.lambda_z_id*dist(s_BA, z_BA)
            if self.lambda_cross:
                loss_rec['z_cross'] = self.lambda_z_id*dist(s_cross, s_A)
        if self.lambda_sample:
            loss_rec['sample'] = self.lambda_sample*dist(z_BA_im, z_BA_im_rec)
        
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
        gradnorm_G = 0
        if optimizer is not None and isinstance(loss_G, torch.Tensor):
            loss_G.mean().backward()
            optimizer['G'].step()
            gradnorm_G = grad_norm(self)
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_G',             loss_G),
            ('l_D',             loss_D),
            ('l_DA',            _reduce([loss_disc['A']])),
            ('l_DB',            _reduce([loss_disc['B']])),
            ('l_gen_AB',        _reduce([loss_gen['AB']])),
            ('l_gen_BA',        _reduce([loss_gen['BA']])),
            ('l_gen_cross',     _reduce([loss_gen['cross']])),
            ('l_rec_sample',    _reduce([loss_rec['sample']])),
            ('l_rec',           _reduce([loss_rec['BB']])),
            ('l_rec_z',         _reduce([_cat([loss_rec['z_BA'],
                                               loss_rec['z_cross']], dim=1)])),
            ('l_rec_z_BA',      _reduce([loss_rec['z_BA']])),
            ('l_rec_z_cross',   _reduce([loss_rec['z_cross']])),
            ('l_cyc',           _reduce([loss_cyc])),
            ('l_seg',           _reduce([loss_seg])),
            ('l_gradnorm_G',    gradnorm_G),
            ('l_gradnorm_D',    gradnorm_D),
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
