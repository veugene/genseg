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
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, encoder, decoder, segmenter, disc_A, disc_B,
                 shape_sample, disc_cross=None, loss_rec=mae, loss_seg=None,
                 loss_gan='hinge', num_disc_updates=1, relativistic=False,
                 grad_penalty=None, disc_clip_norm=None, lambda_disc=1,
                 lambda_x_id=10, lambda_z_id=1, lambda_seg=1, lambda_cross=0,
                 lambda_cyc=0, lambda_sample=1, sample_image_space=False,
                 sample_decoder=None, rng=None):
        super(segmentation_model, self).__init__()
        lambdas = {
            'lambda_disc'       : lambda_disc,
            'lambda_x_id'       : lambda_x_id,
            'lambda_z_id'       : lambda_z_id,
            'lambda_seg'        : lambda_seg,
            'lambda_cross'      : lambda_cross,
            'lambda_cyc'        : lambda_cyc,
            'lambda_sample'     : lambda_sample
            }
        kwargs = {
            'rng'               : rng if rng else np.random.RandomState(),
            'encoder'           : encoder,
            'decoder'           : decoder,
            'segmenter'         : [segmenter],  # Separate params.
            'disc_A'            : disc_A,
            'disc_B'            : disc_B,
            'disc_C'            : disc_cross if disc_cross else disc_A,
            'shape_sample'      : shape_sample,
            'loss_rec'          : loss_rec,
            'loss_seg'          : loss_seg if loss_seg else dice_loss(),
            'loss_gan'          : loss_gan,
            'num_disc_updates'  : num_disc_updates,
            'relativistic'      : relativistic,
            'grad_penalty'      : grad_penalty,
            'disc_clip_norm'    : disc_clip_norm,
            'sample_image_space': sample_image_space,
            'sample_decoder'    : sample_decoder,
            'gan_objective'     : gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)
            }
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['encoder', 'decoder', 'shape_sample',
                        'sample_image_space', 'sample_decoder', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_A', 'disc_B', 'disc_C']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D =_loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_A', 'disc_B', 'disc_C', 'loss_rec']
        kwargs_G = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_G])
        self._loss_G = _loss_G(**kwargs_G, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_G = nn.DataParallel(self._loss_G, output_device=-1)
        
    def forward(self, x_A, x_B, mask=None, optimizer=None, disc=None,
                rng=None):
        # Compute gradients and update?
        do_updates_bool = True if optimizer is not None else False
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            visible, hidden = self._forward(x_A, x_B, rng=rng)
        
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc:
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    loss_disc = self._loss_D(x_A=x_A,
                                             x_B=x_B,
                                             x_BA=visible['x_BA'],
                                             x_AB=visible['x_AB'],
                                             x_cross=visible['x_cross'])
                    loss_D = _reduce(sum(loss_disc.values()))
                # Update discriminator.
                if do_updates_bool:
                    optimizer['D'].zero_grad()
                    loss_D.mean().backward()
                    if self.disc_clip_norm:
                        nn.utils.clip_grad_norm_(self.disc_A.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(self.disc_B.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(self.disc_C.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    optimizer['D'].step()
                    gradnorm_D = grad_norm(self.disc_A)+grad_norm(self.disc_B)
                    if self.disc_C is not None:
                        gradnorm_D = gradnorm_D+grad_norm(self.disc_C)
        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            losses_G = self._loss_G(x_A=x_A,
                                    x_B=x_B,
                                    x_AB=visible['x_AB'],
                                    x_BA=visible['x_BA'],
                                    x_BB=visible['x_BB'],
                                    x_cross=visible['x_cross'],
                                    x_cross_A=visible['x_cross_A'],
                                    x_AM=visible['x_AM'],
                                    **hidden)
        
        # Compute segmentation loss outside of DataParallel modules,
        # avoiding various issues:
        # - scatter of small batch sizes can lead to empty tensors
        # - tracking mask indices is very messy
        # - Dice loss reduced before being returned; then, averaged over GPUs
        mask_packed = x_AM_packed = None
        if mask is not None:
            # Prepare a mask Tensor without None entries.
            mask_indices = [i for i, m in enumerate(mask) if m is not None]
            x_AM_packed = visible['x_AM'][mask_indices]
            mask_packed = np.array([mask[i] for i in mask_indices])
            mask_packed = Variable(torch.from_numpy(mask_packed))
            if torch.cuda.device_count()==1:
                # `DataParallel` does not respect `output_device` when
                # there is only one GPU. So it returns outputs on GPU rather
                # than CPU, as requested. When this happens, putting mask
                # on GPU allows all values to stay on one device.
                mask_packed = mask_packed.cuda()
        loss_seg = 0.
        if self.lambda_seg and mask_packed is not None and len(mask_packed):
            loss_seg = self.lambda_seg*self.loss_seg(x_AM_packed, mask_packed)
        
        # Include segmentation loss with generator losses and update.
        losses_G['l_seg'] = _reduce([loss_seg])
        loss_G = losses_G['l_G']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            optimizer['G'].zero_grad()
            loss_G.mean().backward()
            optimizer['G'].step()
            gradnorm_G = grad_norm(self)
        
        # Compile ouputs.
        outputs = OrderedDict()
        outputs['x_M'] = mask_packed
        outputs.update(visible)
        outputs['x_AM'] = x_AM_packed
        outputs.update(losses_G)
        outputs['l_D']  = loss_D
        outputs['l_DA'] = _reduce([loss_disc['A']])
        outputs['l_DB'] = _reduce([loss_disc['B']])
        outputs['l_DC'] = _reduce([loss_disc['C']])
        
        return outputs


class _forward(nn.Module):
    def __init__(self, encoder, decoder, shape_sample, lambda_disc=1,
                 lambda_x_id=10, lambda_z_id=1, lambda_seg=1, lambda_cross=0,
                 lambda_cyc=0, lambda_sample=1, sample_image_space=False,
                 sample_decoder=None, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder            = encoder
        self.decoder            = decoder
        self.shape_sample       = shape_sample
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_cross       = lambda_cross
        self.lambda_cyc         = lambda_cyc
        self.lambda_sample      = lambda_sample
        self.sample_image_space = sample_image_space
        self.sample_decoder     = sample_decoder
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self.shape_sample).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        ret = ret.to(torch.cuda.current_device())
        return ret
    
    def forward(self, x_A, x_B, rng=None):
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
            if z_BA.size(1)<s_B.size(1):
                # When the sample at the bottleneck has fewer features (n) 
                # than the code resulting from the encoder (N), concatenate 
                # the first N-n features from the code to the n features in 
                # the sample.
                z_BA = torch.cat([s_B[:,:s_B.size(1)-z_BA.size(1)],
                                  z_BA], dim=1)
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
        if self.lambda_seg:
            z_AM, skip_AM = self.decoder(s_A, skip_info=skip_A)
            x_AM = self.segmenter[0](z_AM, skip_info=skip_AM)
        
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
        visible = OrderedDict((
            ('x_AM',               x_AM),
            ('x_A',                x_A),
            ('x_AB',               x_AB),
            ('x_AB_residual',      x_AB_residual),
            ('x_AA',               x_AA)
            ('x_B',                x_B),
            ('x_BA',               x_BA),
            ('x_BA_residual',      x_BA_residual),
            ('x_BB',               x_BB),
            ('x_cross_A',          x_cross_A),
            ('x_cross_A_residual', x_cross_A_residual),
            ('x_cross',            x_cross),
            ('x_cross_residual',   x_cross_residual)
            ))
        hidden = {
            's_BA'       : s_BA,
            'z_BA'       : z_BA,
            's_cross'    : s_cross,
            's_A'        : s_A,
            'z_BA_im'    : z_BA_im,
            'z_BA_im_rec': z_BA_im_rec}
        return visible, hidden


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, disc_C,
                 lambda_disc=1, lambda_x_id=10, lambda_z_id=1, lambda_seg=1,
                 lambda_cross=0, lambda_cyc=0, lambda_sample=1):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.disc_C             = disc_C
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_cross       = lambda_cross
        self.lambda_cyc         = lambda_cyc
        self.lambda_sample      = lambda_sample
    
    def forward(self, x_A, x_B, x_BA, x_AB, x_cross):
        loss_disc = OrderedDict()
        loss_disc_A = self._gan.D(self.disc_A,
                                  fake=x_BA.detach(), real=x_A)
        loss_disc_B = self._gan.D(self.disc_B,
                                  fake=x_AB.detach(), real=x_B)
        loss_disc['A'] = self.lambda_disc*loss_disc_A
        loss_disc['B'] = self.lambda_disc*loss_disc_B
        if self.lambda_cross:
            loss_disc_C = self._gan.D(self.disc_C,
                                      fake=x_cross.detach(), real=x_A)
            loss_disc['C'] = ( self.lambda_disc*self.lambda_cross
                              *loss_disc_C)
        return loss_disc


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, disc_C, loss_rec=mae,
                 lambda_disc=1, lambda_x_id=10, lambda_z_id=1, lambda_seg=1,
                 lambda_cross=0, lambda_cyc=0, lambda_sample=1):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.disc_C             = disc_C
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_seg         = lambda_seg
        self.lambda_cross       = lambda_cross
        self.lambda_cyc         = lambda_cyc
        self.lambda_sample      = lambda_sample
    
    def forward(self, x_A, x_B, x_AB, x_BA, x_BB, x_cross, x_cross_A, x_AM,
                s_BA, z_BA, s_cross, s_A, z_BA_im, z_BA_im_rec):
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            loss_gen['AB'] = self.lambda_disc*self._gan.G(self.disc_B,
                                                          fake=x_AB, real=x_B)
            loss_gen['BA'] = self.lambda_disc*self._gan.G(self.disc_A,
                                                          fake=x_BA, real=x_A)
        if self.lambda_disc and self.lambda_cross:
            loss_gen['C'] = ( self.lambda_disc*self.lambda_cross
                             *self._gan.G(self.disc_C,
                                          fake=x_cross, real=x_A))
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        if self.lambda_x_id:
            loss_rec['BB'] = self.lambda_x_id*self.loss_rec(x_BB, x_B)
        if self.lambda_z_id:
            loss_rec['z_BA'] = self.lambda_z_id*self.loss_rec(s_BA, z_BA)
            if self.lambda_cross:
                loss_rec['z_cross'] = self.lambda_z_id*self.loss_rec(s_cross,
                                                                     s_A)
        if self.lambda_sample:
            loss_rec['sample'] = self.lambda_sample*self.loss_rec(z_BA_im,
                                                                  z_BA_im_rec)
        
        # Cross cycle consistency loss.
        loss_cyc = 0
        if self.lambda_cyc:
            loss_cyc = self.lambda_cyc*self.loss_rec(x_cross_A, x_A)
        
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',             loss_G),
            ('l_gen_AB',        _reduce([loss_gen['AB']])),
            ('l_gen_BA',        _reduce([loss_gen['BA']])),
            ('l_gen_cross',     _reduce([loss_gen['C']])),
            ('l_rec_sample',    _reduce([loss_rec['sample']])),
            ('l_rec',           _reduce([loss_rec['BB']])),
            ('l_rec_z',         _reduce([_cat([loss_rec['z_BA'],
                                               loss_rec['z_cross']], dim=1)])),
            ('l_rec_z_BA',      _reduce([loss_rec['z_BA']])),
            ('l_rec_z_cross',   _reduce([loss_rec['z_cross']])),
            ('l_cyc',           _reduce([loss_cyc])),
            ))
        return losses
