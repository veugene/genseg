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
    def __init__(self, encoder, decoder, segmenter, disc_A, disc_B,
                 shape_sample, disc_cross=None, preprocessor=None,
                 postprocessor=None, loss_rec=mae, loss_seg=None,
                 loss_gan='hinge', num_disc_updates=1,  relativistic=False,
                 grad_penalty=None, disc_clip_norm=None, lambda_disc=1,
                 lambda_x_id=10, lambda_z_id=1, lambda_seg=1, lambda_cross=0,
                 lambda_cyc=0,lambda_sample=1, sample_image_space=False,
                 sample_decoder=None, rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder          = encoder
        self.decoder          = decoder
        self.segmenter        = [segmenter]     # separate params
        if disc_cross is None:
            disc_cross = disc_A
        self.disc = {'A'    :     disc_A,
                     'B'    :     disc_B,
                     'C'    :     disc_cross}   # separate params
        self.shape_sample       = shape_sample
        self.preprocessor       = preprocessor
        self.postprocessor      = postprocessor
        self.loss_rec           = loss_rec
        self.loss_seg           = loss_seg if loss_seg else dice_loss()
        self.loss_gan           = loss_gan
        self.num_disc_updates   = num_disc_updates
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
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self.shape_sample).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _zcat(self, code, sample):
        if sample.size(1)==code.size(1):
            return sample
        # When the sample at the bottleneck has fewer features (n) than the
        # code resulting from the encoder (N), concatenate the first N-n
        # features from the code to the n features in the sample.
        return torch.cat([code[:,:code.size(1)-sample.size(1)], sample], dim=1)
    
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
        penc = self.preprocessor  if self.preprocessor  else lambda x:x
        pdec = self.postprocessor if self.postprocessor else lambda x:x
        s_A, skip_A = self.encoder(penc(x_A))
        x_AB = pdec(penc(x_A) - self.decoder(s_A, skip_info=skip_A))
        return x_AB
    
    def translate_BA(self, x_B, rng=None):
        penc = self.preprocessor  if self.preprocessor  else lambda x:x
        pdec = self.postprocessor if self.postprocessor else lambda x:x
        batch_size = len(x_B)
        s_B, skip_B = self.encoder(penc(x_B))
        if self.sample_image_space:
            z_BA_im = self._z_sample(batch_size, rng=rng)
            z_BA, _ = self.encoder(penc(z_BA_im))
        else:
            z_BA    = self._z_sample(batch_size, rng=rng)
        z_BA = self._zcat(s_B, z_BA)
        x_BA = pdec(penc(x_B) + self.decoder(z_BA, skip_info=skip_B))
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_A)
        s_A, skip_A = self.encoder(penc(x_A))
        x_AM = self.segmenter[0](
            torch.cat([self.decoder(s_A, skip_info=skip_A), x_A], dim=1))
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
        
        # Aliases.
        penc = self.preprocessor  if self.preprocessor  else lambda x:x
        pdec = self.postprocessor if self.postprocessor else lambda x:x
        x_A_ = penc(x_A)
        x_B_ = penc(x_B)
        
        # Encode inputs.
        s_A, skip_A = self.encoder(x_A_)
        only_seg = True
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id):
                s_B, skip_B = self.encoder(x_B_)
                only_seg = False
        
        # Translate.
        x_AB = x_BA = x_AB_residual = x_BA_residual = None
        if self.lambda_disc or self.lambda_z_id:
            if self.sample_image_space:
                z_BA_im = self._z_sample(batch_size, rng=rng)
                z_BA, _ = self.encoder(penc(z_BA_im))
            else:
                z_BA    = self._z_sample(batch_size, rng=rng)
            z_BA = self._zcat(s_B, z_BA)
            x_BA_residual = self.decoder(z_BA, skip_info=skip_B)
            x_AB_residual = self.decoder(s_A,  skip_info=skip_A)
            x_BA = pdec(x_B_ + x_BA_residual)              # (+)
            x_AB = pdec(x_A_ - x_AB_residual)              # (-)
        x_cross = x_cross_residual = None
        if self.lambda_disc and self.lambda_cross:
            x_cross_residual = self.decoder(s_A, skip_info=skip_B)
            x_cross = pdec(x_B_ + x_cross_residual)        # (+)
                
        # Reconstruct input.
        x_AA = x_BB = z_BA_im_rec = None
        if self.lambda_x_id:
            x_BB = pdec(x_B_ - self.decoder(s_B, skip_info=skip_B))  # (-)
            if self.preprocessor is not None or self.postprocessor is not None:
                x_AA = pdec(x_A_)
        if self.sample_image_space and self.sample_decoder is not None:
            z_BA_im_rec = self.sample_decoder(z_BA)
        
        # Segment.
        x_AM = None
        if self.lambda_seg and mask is not None and len(mask)>0:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            skip_A_filtered = [s[mask_indices] for s in skip_A]
            f = torch.cat([self.decoder(s_A[mask_indices],
                                        skip_info=skip_A_filtered),
                           x_A[mask_indices]], dim=1)
            x_AM = self.segmenter[0](f)
        
        # Reconstruct latent code.
        if self.lambda_z_id:
            s_BA, skip_BA = self.encoder(penc(x_BA))
            if self.lambda_cross:
                s_cross, skip_cross = self.encoder(penc(x_cross))
        
        # Cycle.
        x_cross_A = x_cross_A_residual = None
        if self.lambda_cyc:
            s_AB, skip_AB = self.encoder(penc(x_AB))
            x_cross_A_residual = self.decoder(s_cross, skip_info=skip_AB)
            x_cross_A = pdec(penc(x_AB) + x_cross_A_residual)   # (+)
        
        # Discriminator losses.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc:
            for i in range(self.num_disc_updates):
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
                loss_D = _reduce([ loss_disc['A']+loss_disc['B']
                                  +loss_disc['C']])
                if optimizer is not None:
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
            if self.preprocessor is not None or self.postprocessor is not None:
                loss_rec['AA'] = self.lambda_x_id*dist(x_AA, x_A)
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
            optimizer['S'].step()
            gradnorm_G = grad_norm(self)
        
        # Don't display residuals if they have more channels than the images.
        ch = x_A.size(1)
        if x_AB_residual is not None and x_AB_residual.size(1)>ch:
            x_AB_residual = None
        if x_BA_residual is not None and x_BA_residual.size(1)>ch:
            x_BA_residual = None
        if x_cross_residual is not None and x_cross_residual.size(1)>ch:
            x_cross_residual = None
        if x_cross_A_residual is not None and x_cross_A_residual.size(1)>ch:
            x_cross_A_residual = None
        
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
            ('out_M',           mask),
            ('out_AM',          x_AM),
            ('out_A',           x_A),
            ('out_AB',          x_AB),
            ('out_AB_res',      x_AB_residual),
            ('out_AA',          x_AA),
            ('out_B',           x_B),
            ('out_BA',          x_BA),
            ('out_BA_res',      x_BA_residual),
            ('out_BB',          x_BB),
            ('out_cross',       x_cross),
            ('out_cross_res',   x_cross_residual),
            ('out_cross_A',     x_cross_A),
            ('out_cross_A_res', x_cross_A_residual)))
        return outputs
