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
from .common.mine import mine


def _reduce(loss):
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    return sum([_mean(v) for v in loss])


class segmentation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, encoder, decoder_common, decoder_residual, segmenter,
                 disc_A, disc_B, shape_sample, mutual_information=None,
                 loss_rec=mae, loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, relativistic=False, grad_penalty=None,
                 disc_clip_norm=None, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_f_id=1, lambda_seg=1, lambda_cyc=0,
                 lambda_mi=1, rng=None):
        super(segmentation_model, self).__init__()
        lambdas = {
            'lambda_disc'       : lambda_disc,
            'lambda_x_id'       : lambda_x_id,
            'lambda_z_id'       : lambda_z_id,
            'lambda_f_id'       : lambda_f_id,
            'lambda_seg'        : lambda_seg,
            'lambda_cyc'        : lambda_cyc,
            'lambda_mi'         : lambda_mi,
            }
        kwargs = {
            'rng'               : rng if rng else np.random.RandomState(),
            'encoder'           : encoder,
            'decoder_common'    : decoder_common,
            'decoder_residual'  : decoder_residual,
            'mutual_information': mutual_information,
            'shape_sample'      : shape_sample,
            'loss_rec'          : loss_rec,
            'loss_seg'          : loss_seg if loss_seg else dice_loss(),
            'loss_gan'          : loss_gan,
            'num_disc_updates'  : num_disc_updates,
            'relativistic'      : relativistic,
            'grad_penalty'      : grad_penalty,
            'disc_clip_norm'    : disc_clip_norm,
            'gan_objective'     : gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)
            }
        self.separate_networks = {
            'segmenter'         : segmenter,
            'mi_estimator'      : None,
            'disc_A'            : disc_A,
            'disc_B'            : disc_B,
            }
        kwargs.update(lambdas)
        for key, val in kwargs.items():
            setattr(self, key, val)
        # Set up mutual information estimator.
        if mutual_information is not None:
            self.separate_networks['mi_estimator'] = mine(mutual_information,
                                                          rng=self.rng)
        # Separate networks not stored directly as attributes.
        # -> Separate parameters, separate optimizers.
        kwargs.update(self.separate_networks)
        
        # Module to compute all network outputs (except discriminator) on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_forward = ['encoder', 'decoder_common', 'decoder_residual',
                        'segmenter', 'shape_sample', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_A', 'disc_B', 'mi_estimator']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D =_loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_A', 'disc_B', 'mi_estimator',
                  'loss_rec']
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
            visible, hidden, intermediates = self._forward(x_A, x_B, rng=rng)
        
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc:
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    loss_disc, loss_mi_est = self._loss_D(x_A=x_A,
                                                          x_B=x_B,
                                                          x_BA=visible['x_BA'],
                                                          x_AB=visible['x_AB'],
                                                          c_A=hidden['c_A'],
                                                          u_A=hidden['u_A'],
                                                          c_BA=hidden['c_BA'],
                                                          u_BA=hidden['u_BA'])
                    loss_D = _reduce(loss_disc.values())
                # Update discriminator.
                disc_A = self.separate_networks['disc_A']
                disc_B = self.separate_networks['disc_B']
                if do_updates_bool:
                    optimizer['D'].zero_grad()
                    loss_D.mean().backward()
                    if self.disc_clip_norm:
                        nn.utils.clip_grad_norm_(disc_A.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_B.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    optimizer['D'].step()
                    gradnorm_D = grad_norm(disc_A)+grad_norm(disc_B)
                # Update MI estimator.
                mi_estimator = self.separate_networks['mi_estimator']
                if do_updates_bool and mi_estimator is not None:
                    optimizer['E'].zero_grad()
                    _reduce([loss_mi_est['A'],
                             loss_mi_est['BA']]).mean().backward()
                    optimizer['E'].step()
        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            losses_G = self._loss_G(x_AM=visible['x_AM'],
                                    x_A=x_A,
                                    x_AB=visible['x_AB'],
                                    x_AA=visible['x_AA'],
                                    x_B=x_B,
                                    x_BA=visible['x_BA'],
                                    x_BB=visible['x_BB'],
                                    x_BAB=visible['x_BAB'],
                                    **hidden,
                                    **intermediates)
        
        # Compute segmentation loss outside of DataParallel modules,
        # avoiding various issues:
        # - scatter of small batch sizes can lead to empty tensors
        # - tracking mask indices is very messy
        # - Dice loss reduced before being returned; then, averaged over GPUs
        mask_packed = x_AM_packed = None
        if mask is not None:
            # Prepare a mask Tensor without None entries.
            mask_indices = [i for i, m in enumerate(mask) if m is not None]
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
            x_AM_packed = visible['x_AM'][mask_indices]
            loss_seg = self.lambda_seg*self.loss_seg(x_AM_packed, mask_packed)
        
        # Include segmentation loss with generator losses and update.
        losses_G['l_seg'] = _reduce([loss_seg])
        losses_G['l_G'] += losses_G['l_seg']
        loss_G = losses_G['l_G']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            if 'S' in optimizer:
                optimizer['S'].zero_grad()
            optimizer['G'].zero_grad()
            loss_G.mean().backward()
            optimizer['G'].step()
            if 'S' in optimizer:
                optimizer['S'].step()
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
        outputs['l_gradnorm_D'] = gradnorm_D
        outputs['l_gradnorm_G'] = gradnorm_G
        
        return outputs


class _forward(nn.Module):
    def __init__(self, encoder, decoder_common, decoder_residual, segmenter,
                 shape_sample, lambda_disc=1, lambda_x_id=10, lambda_z_id=1,
                 lambda_f_id=1, lambda_seg=1, lambda_cyc=0, lambda_mi=1,
                 rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder            = encoder
        self.decoder_common     = decoder_common
        self.decoder_residual   = decoder_residual
        self.segmenter          = [segmenter]   # Separate params.
        self.shape_sample       = shape_sample
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
    
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
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id):
                s_B, skip_B = self.encoder(x_B)
        
        # Helper function for summing either two tensors or pairs of tensors
        # across two lists of tensors.
        def add(a, b):
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                return a+b
            else:
                assert not isinstance(a, torch.Tensor)
                assert not isinstance(b, torch.Tensor)
                return [elem_a+elem_b for elem_a, elem_b in zip(a, b)]
        
        # Helper function to split an output into the final image output
        # tensor and a list of intermediate tensors.
        def unpack(x):
            x_list = None
            if not isinstance(x, torch.Tensor):
                x, x_list = x[-1], x[:-1]
            return x, x_list
        
        # A->(B, dA)->A
        x_AB = x_AB_residual = X_AA = x_AA_list = c_A = u_A = None
        if (self.lambda_seg
         or self.lambda_disc or self.lambda_x_id or self.lambda_z_id):
            x_AB_residual, skip_AM = self.decoder_residual(s_A,
                                                           skip_info=skip_A)
        if self.lambda_disc or self.lambda_x_id or self.lambda_z_id:
            c_A, u_A = torch.split(s_A, [s_A.size(1)-self.shape_sample[0],
                                         self.shape_sample[0]], dim=1)
            x_AB, _ = self.decoder_common(c_A, skip_info=skip_A)
            x_AA = add(x_AB, x_AB_residual)
            
            # Unpack.
            x_AA, x_AA_list = unpack(x_AA)
            x_AB, _         = unpack(x_AB)
            x_AB_residual, _= unpack(x_AB_residual)
        
        # B->(B, dA)->A
        x_BA = x_BA_residual = x_BB = z_BA = x_BB_list = None
        if self.lambda_disc or self.lambda_x_id or self.lambda_z_id:
            u_BA = self._z_sample(batch_size, rng=rng)
            c_B  = s_B[:,:s_B.size(1)-u_BA.size(1)]
            z_BA = torch.cat([c_B, u_BA], dim=1)
            x_BB, _ = self.decoder_common(c_B, skip_info=skip_B)
            x_BA_residual, _ = self.decoder_residual(z_BA, skip_info=skip_B)
            x_BA = add(x_BB, x_BA_residual)
            
            # Unpack.
            x_BA, _         = unpack(x_BA)
            x_BB, x_BB_list = unpack(x_BB)
            x_BA_residual, _= unpack(x_BA_residual)
        
        # Segment.
        x_AM = None
        if self.lambda_seg:
            if self.segmenter[0] is not None:
                x_AM = self.segmenter(s_A, skip_info=skip_AM)
            else:
                # Re-use residual decoder in mode 1.
                x_AM = self.decoder_residual(s_A, skip_info=skip_AM, mode=1)
                x_AM, _ = unpack(x_AM)
        
        # Reconstruct latent codes.
        s_BA = s_AA = c_AB = c_BB = None
        if self.lambda_z_id or self.lambda_cyc:
            s_BA, skip_BA = self.encoder(x_BA)
            s_AA, _       = self.encoder(x_AA)
            s_AB, _       = self.encoder(x_AB)
            s_BB, _       = self.encoder(x_BB)
            c_AB = s_AB[:,:s_AB.size(1)-self.shape_sample[0]]
            c_BB = s_BB[:,:s_BB.size(1)-self.shape_sample[0]]
        
        # Cycle.
        x_BAB = c_BA = u_BA = None
        if self.lambda_cyc:
            c_BA, u_BA = torch.split(s_BA, [s_BA.size(1)-self.shape_sample[0],
                                            self.shape_sample[0]], dim=1)
            x_BAB, _ = self.decoder_common(c_BA, skip_info=skip_BA)
            x_BAB, _ = unpack(x_BAB)
        
        # Compile outputs and return.
        visible = OrderedDict((
            ('x_AM',          x_AM),
            ('x_A',           x_A),
            ('x_AB',          x_AB),
            ('x_AB_residual', x_AB_residual),
            ('x_AA',          x_AA),
            ('x_B',           x_B),
            ('x_BA',          x_BA),
            ('x_BA_residual', x_BA_residual),
            ('x_BB',          x_BB),
            ('x_BAB',         x_BAB),
            ))
        hidden = {
            's_BA'          : s_BA,
            's_AA'          : s_AA,
            'c_AB'          : c_AB,
            'c_BB'          : c_BB,
            'z_BA'          : z_BA,
            's_A'           : s_A,
            'c_A'           : c_A,
            'u_A'           : u_A,
            'c_B'           : c_B,
            'c_BA'          : c_BA,
            'u_BA'          : u_BA}
        intermediates = {
            'x_AA_list'     : x_AA_list,
            'x_BB_list'     : x_BB_list,
            'skip_A'        : skip_A,
            'skip_B'        : skip_B}
        return visible, hidden, intermediates


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, mi_estimator=None,
                 lambda_disc=1, lambda_x_id=10, lambda_z_id=1, lambda_f_id=1,
                 lambda_seg=1, lambda_cyc=0, lambda_mi=1):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.mi_estimator       = mi_estimator
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
    
    def forward(self, x_A, x_B, x_BA, x_AB, c_A, u_A, c_BA, u_BA):
        # If outputs are lists, get the last item (image).
        if not isinstance(x_BA, torch.Tensor):
            x_BA = x_BA[-1]
        if not isinstance(x_AB, torch.Tensor):
            x_AB = x_AB[-1]
        
        # Discriminators.
        loss_disc = OrderedDict()
        loss_disc_A = self._gan.D(self.disc_A,
                                  fake=x_BA.detach(), real=x_A)
        loss_disc_B = self._gan.D(self.disc_B,
                                  fake=x_AB.detach(), real=x_B)
        loss_disc['A'] = self.lambda_disc*loss_disc_A
        loss_disc['B'] = self.lambda_disc*loss_disc_B
        
        # Mutual information estimate.
        loss_mi_est = defaultdict()
        if self.mi_estimator is not None:
            loss_mi_est['A'] = self.mi_estimator(c_A.detach(), u_A.detach())
            if self.lambda_cyc:
                loss_mi_est['BA'] = self.mi_estimator(c_BA.detach(),
                                                      u_BA.detach())
        
        return loss_disc, loss_mi_est


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, mi_estimator=None,
                 loss_rec=mae, lambda_disc=1, lambda_x_id=10, lambda_z_id=1,
                 lambda_f_id=1, lambda_seg=1, lambda_cyc=0, lambda_mi=1):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.mi_estimator       = mi_estimator
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
    
    def forward(self, x_AM, x_A, x_AB, x_AA, x_B, x_BA, x_BB, x_BAB,
                s_BA, s_AA, c_AB, c_BB, z_BA, s_A, c_A, u_A, c_B, c_BA, u_BA,
                x_AA_list, x_BB_list, skip_A, skip_B):
        # Mutual information loss for generator.
        loss_mi_gen = defaultdict(int)
        if self.mi_estimator is not None:
            loss_mi_gen['A']  = -self.lambda_mi*self.mi_estimator(c_A, u_A)
            loss_mi_gen['BA'] = -self.lambda_mi*self.mi_estimator(c_BA, u_BA)
        
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            loss_gen['AB'] = self.lambda_disc*self._gan.G(self.disc_B,
                                                          fake=x_AB, real=x_B)
            loss_gen['BA'] = self.lambda_disc*self._gan.G(self.disc_A,
                                                          fake=x_BA, real=x_A)
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        if self.lambda_x_id:
            loss_rec['AA'] = self.lambda_x_id*self.loss_rec(x_AA, x_A)
            loss_rec['BB'] = self.lambda_x_id*self.loss_rec(x_BB, x_B)
        if self.lambda_x_id and self.lambda_cyc:
            loss_rec['BB'] += self.lambda_cyc*self.loss_rec(x_BAB, x_B)
        if self.lambda_z_id:
            loss_rec['z_BA'] = self.lambda_z_id*self.loss_rec(s_BA, z_BA)
            loss_rec['z_AB'] = self.lambda_z_id*self.loss_rec(c_AB, c_A)
            loss_rec['z_AA'] = self.lambda_z_id*self.loss_rec(s_AA, s_A)
            loss_rec['z_BB'] = self.lambda_z_id*self.loss_rec(c_BB, c_B)
        
        # Reconstruction of intermediate features.
        if self.lambda_f_id:
            loss_rec['AA'] = _reduce([loss_rec['AA']])
            loss_rec['BB'] = _reduce([loss_rec['BB']])
            for s, t in zip(x_AA_list, skip_A[::-1]):
                loss_rec['AA'] += _reduce([ self.lambda_f_id
                                           *self.loss_rec(s, t)])
            for s, t in zip(x_BB_list, skip_B[::-1]):
                loss_rec['BB'] += _reduce([ self.lambda_f_id
                                           *self.loss_rec(s, t)])
        
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values())
                  +_reduce(loss_mi_gen.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_AB',      _reduce([loss_gen['AB']])),
            ('l_gen_BA',      _reduce([loss_gen['BA']])),
            ('l_rec',         _reduce([loss_rec['AA'], loss_rec['BB']])),
            ('l_rec_AA',      _reduce([loss_rec['AA']])),
            ('l_rec_BB',      _reduce([loss_rec['BB']])),
            ('l_rec_c',       _reduce([loss_rec['z_AB'], loss_rec['z_BB']])),
            ('l_rec_s',       _reduce([loss_rec['z_BA'], loss_rec['z_AA']])),
            ('l_rec_z_BA',    _reduce([loss_rec['z_BA']])),
            ('l_rec_z_AB',    _reduce([loss_rec['z_AB']])),
            ('l_rec_z_AA',    _reduce([loss_rec['z_AA']])),
            ('l_rec_z_BB',    _reduce([loss_rec['z_BB']])),
            ('l_mi',          _reduce([loss_mi_gen['A'], loss_mi_gen['BA']])),
            ('l_mi_A',        _reduce([loss_mi_gen['A']])),
            ('l_mi_BA',       _reduce([loss_mi_gen['BA']]))
            ))
        return losses
