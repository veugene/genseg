from collections import (defaultdict,
                         OrderedDict)
from contextlib import nullcontext
import functools
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.cuda.amp import autocast
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
from .common.network.basic import grad_norm
from .common.losses import (bce,
                            cce,
                            dist_ratio_mse_abs,
                            gan_objective,
                            mae,
                            mse)
from .common.mine import mine


def clear_grad(optimizer):
    # Sets `grad` to None instead of zeroing it.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = None


def _reduce(loss):
    # Reduce torch tensors with dim > 1 with a mean on all but the first
    # (batch) dimension. Else, return as is.
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    if len(loss)==0: return 0
    return sum([_mean(v) for v in loss])


def _cce(p, t):
    # Cross entropy loss that can handle multi-scale classifiers.
    # 
    # If p is a list, process every element (and reduce to batch dim).
    # Each tensor is reduced by `mean` and reduced tensors are averaged
    # together.
    # (For multi-scale classifiers. Each scale is given equal weight.)
    if not isinstance(p, torch.Tensor):
        return sum([_cce(elem, t) for elem in p])/float(len(p))
    # Convert target list to torch tensor (batch_size, 1, 1, 1).
    t = torch.Tensor(t).reshape(-1,1,1).expand(-1,p.size(2),p.size(3)).long()
    if p.is_cuda:
        t = t.to(p.device)
    # Cross-entropy.
    out = F.cross_entropy(p, t)
    # Return if no dimensions beyond batch dim.
    if out.dim()<=1:
        return out
    # Else, return after reducing to batch dim.
    return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.


def autocast_if_needed():
    # Decorator. If the method's object has a scaler, use the pytorch
    # autocast context; else, run the method without any context.
    def decorator(method):
        @functools.wraps(method)
        def context_wrapper(cls, *args, **kwargs):
            if cls.scaler is not None:
                with torch.cuda.amp.autocast():
                    return method(cls, *args, **kwargs)
            return method(cls, *args, **kwargs)
        return context_wrapper
    return decorator


class segmentation_model(nn.Module):
    """
    Interface wrapper around the `DataParallel` parts of the model.
    """
    def __init__(self, encoder, decoder_common, decoder_residual, segmenter,
                 disc_A, disc_B, shape_sample, mutual_information=None,
                 decoder_autoencode=None, classifier_A=None, classifier_B=None,
                 scaler=None, loss_rec=mae, loss_seg=None, loss_gan='hinge',
                 num_disc_updates=1, relativistic=False, grad_penalty=None,
                 disc_clip_norm=None,gen_clip_norm=None,  lambda_disc=1,
                 lambda_x_ae=10, lambda_x_id=10, lambda_z_id=1, lambda_f_id=1,
                 lambda_seg=1, lambda_cyc=0,lambda_mi=0, lambda_slice=0.,
                 debug_ac_gan=False,
                 rng=None):
        super(segmentation_model, self).__init__()
        lambdas = OrderedDict((
            ('lambda_disc',       lambda_disc),
            ('lambda_x_ae',       lambda_x_ae),
            ('lambda_x_id',       lambda_x_id),
            ('lambda_z_id',       lambda_z_id),
            ('lambda_f_id',       lambda_f_id),
            ('lambda_seg',        lambda_seg),
            ('lambda_cyc',        lambda_cyc),
            ('lambda_mi',         lambda_mi),
            ('lambda_slice',      lambda_slice)
            ))
        kwargs = OrderedDict((
            ('rng',               rng if rng else np.random.RandomState()),
            ('encoder',           encoder),
            ('decoder_common',    decoder_common),
            ('decoder_residual',  decoder_residual),
            ('decoder_autoencode',decoder_autoencode),
            ('shape_sample',      shape_sample),
            ('scaler',            scaler),
            ('loss_rec',          loss_rec),
            ('loss_seg',          loss_seg if loss_seg else dice_loss()),
            ('loss_gan',          loss_gan),
            ('num_disc_updates',  num_disc_updates),
            ('relativistic',      relativistic),
            ('grad_penalty',      grad_penalty),
            ('gen_clip_norm',     gen_clip_norm),
            ('disc_clip_norm',    disc_clip_norm),
            ('gan_objective',     gan_objective(loss_gan,
                                                relativistic=relativistic,
                                                grad_penalty_real=grad_penalty,
                                                grad_penalty_fake=None,
                                                grad_penalty_mean=0)),
            ('debug_ac_gan',      debug_ac_gan)
            ))
        self.separate_networks = OrderedDict((
            ('segmenter',         segmenter),
            ('mi_estimator',      None),
            ('disc_A',            disc_A),
            ('disc_B',            disc_B),
            ('classifier_A',      classifier_A),
            ('classifier_B',      classifier_B),
            ))
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
                        'decoder_autoencode',  'segmenter', 'shape_sample',
                        'scaler', 'rng']
        kwargs_forward = dict([(key, val) for key, val in kwargs.items()
                               if key in keys_forward])
        self._forward = _forward(**kwargs_forward, **lambdas)
        if torch.cuda.device_count()>1:
            self._forward = nn.DataParallel(self._forward, output_device=-1)
        
        # Module to compute discriminator losses on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_D = ['gan_objective', 'disc_A', 'disc_B', 'mi_estimator',
                  'classifier_A', 'classifier_B', 'scaler', 'debug_ac_gan']
        kwargs_D = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_D])
        self._loss_D = _loss_D(**kwargs_D, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_D = nn.DataParallel(self._loss_D, output_device=-1)
        
        # Module to compute generator updates on GPU.
        # Outputs are placed on CPU when there are multiple GPUs.
        keys_G = ['gan_objective', 'disc_A', 'disc_B', 'mi_estimator',
                  'classifier_A', 'classifier_B', 'scaler', 'loss_rec',
                  'debug_ac_gan']
        kwargs_G = dict([(key, val) for key, val in kwargs.items()
                         if key in keys_G])
        self._loss_G = _loss_G(**kwargs_G, **lambdas)
        if torch.cuda.device_count()>1:
            self._loss_G = nn.DataParallel(self._loss_G, output_device=-1)
    
    def _autocast_if_needed(self):
        # If a scaler is passed, use pytorch gradient autocasting. Else,
        # just use a null context that does nothing.
        if self.scaler is not None:
            context = torch.cuda.amp.autocast()
        else:
            context = nullcontext()
        return context
    
    def forward(self, x_A, x_B, mask=None, class_A=None, class_B=None,
                optimizer=None, rng=None):
        # Compute gradients and update?
        do_updates_bool = True if optimizer is not None else False
        
        # Apply scaler for gradient backprop if it is passed.
        def backward(loss):
            if self.scaler is not None:
                return self.scaler.scale(loss).backward()
            return loss.backward()
        
        # Apply scaler for optimizer step if it is passed.
        def step(optimizer):
            if self.scaler is not None:
                self.scaler.step(optimizer)
            else:
                optimizer.step()
        
        # Compute all outputs.
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                visible, hidden, intermediates = self._forward(x_A, x_B,
                                                               class_A=class_A,
                                                               class_B=class_B,
                                                               rng=rng)
        
        # Evaluate discriminator loss and update.
        loss_disc = defaultdict(int)
        loss_D = gradnorm_D = 0
        if self.lambda_disc:
            if intermediates['x_BA_list'] is None:
                out_BA = visible['x_BA']
            else:
                out_BA = intermediates['x_BA_list']+[visible['x_BA']]
            if intermediates['x_AB_list'] is None:
                out_AB = visible['x_AB']
            else:
                out_AB = intermediates['x_AB_list']+[visible['x_AB']]
            for i in range(self.num_disc_updates):
                # Evaluate.
                with torch.set_grad_enabled(do_updates_bool):
                    with self._autocast_if_needed():
                        loss_disc, loss_slice_est, loss_mi_est = self._loss_D(
                            x_A=x_A,
                            x_B=x_B,
                            class_A=class_A,
                            class_B=class_B,
                            out_BA=out_BA,
                            out_AB=out_AB,
                            c_A=hidden['c_A'],
                            u_A=hidden['u_A'],
                            c_BA=hidden['c_BA'],
                            u_BA=hidden['u_BA'])
                        loss_D = _reduce(loss_disc.values())
                # Update discriminator (and slice classifiers).
                disc_A = self.separate_networks['disc_A']
                disc_B = self.separate_networks['disc_B']
                if do_updates_bool:
                    clear_grad(optimizer['D'])
                    with self._autocast_if_needed():
                        _loss = loss_D.mean()
                    backward(_loss)
                    if self.lambda_slice:
                        with self._autocast_if_needed():
                            _loss = _reduce(loss_slice_est.values()).mean()
                        backward(_loss)
                    if self.disc_clip_norm:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer['D'])
                        nn.utils.clip_grad_norm_(disc_A.parameters(),
                                                 max_norm=self.disc_clip_norm)
                        nn.utils.clip_grad_norm_(disc_B.parameters(),
                                                 max_norm=self.disc_clip_norm)
                    step(optimizer['D'])
                    gradnorm_D = grad_norm(disc_A)+grad_norm(disc_B)
                # Update MI estimator.
                mi_estimator = self.separate_networks['mi_estimator']
                if do_updates_bool and mi_estimator is not None:
                    clear_grad(optimizer['E'])
                    with self._autocast_if_needed():
                        _loss = _reduce([loss_mi_est['A'],
                                         loss_mi_est['BA']]).mean()
                    backward(_loss)
                    step(optimizer['E'])
        
        # Evaluate generator losses.
        gradnorm_G = 0
        with torch.set_grad_enabled(do_updates_bool):
            with self._autocast_if_needed():
                losses_G = self._loss_G(x_AM=visible['x_AM'],
                                        x_A=x_A,
                                        class_A=class_A,
                                        x_AB=visible['x_AB'],
                                        x_AA=visible['x_AA'],
                                        x_B=x_B,
                                        class_B=class_B,
                                        x_BA=visible['x_BA'],
                                        x_BB=visible['x_BB'],
                                        x_BAB=visible['x_BAB'],
                                        x_AA_ae=visible['x_AA_ae'],
                                        x_BB_ae=visible['x_BB_ae'],
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
            with self._autocast_if_needed():
                x_AM_packed = visible['x_AM'][mask_indices]
                print('x_AM_packed')
                print(x_AM_packed.shape)
                print('mask_packed')
                print(mask_packed.shape)
                loss_seg = self.lambda_seg*self.loss_seg(x_AM_packed,
                                                         mask_packed)
        
        # Include segmentation loss with generator losses and update.
        with self._autocast_if_needed():
            losses_G['l_seg'] = _reduce([loss_seg])
            losses_G['l_G'] += losses_G['l_seg']
            loss_G = losses_G['l_G']
        if do_updates_bool and isinstance(loss_G, torch.Tensor):
            if 'S' in optimizer:
                clear_grad(optimizer['S'])
            clear_grad(optimizer['G'])
            with self._autocast_if_needed():
                _loss = loss_G.mean()
            backward(_loss)
            if self.scaler is not None:
                self.scaler.unscale_(optimizer['G'])
                if 'S' in optimizer:
                    self.scaler.unscale_(optimizer['S'])
            if self.gen_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.parameters(),
                                         max_norm=self.gen_clip_norm)
            step(optimizer['G'])
            if 'S' in optimizer:
                step(optimizer['S'])
            gradnorm_G = grad_norm(self)
        
        # Unscale norm.
        if self.scaler is not None and do_updates_bool:
            gradnorm_D /= self.scaler.get_scale()
            gradnorm_G /= self.scaler.get_scale()
        
        # Update scaler.
        if self.scaler is not None and do_updates_bool:
            self.scaler.update()
        
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
                 shape_sample, decoder_autoencode=None, scaler=None,
                 lambda_disc=1, lambda_x_ae=10, lambda_x_id=10, lambda_z_id=1,
                 lambda_f_id=1, lambda_seg=1, lambda_cyc=0, lambda_mi=0,
                 lambda_slice=0, rng=None):
        super(_forward, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder            = encoder
        self.decoder_common     = decoder_common
        self.decoder_residual   = decoder_residual
        self.segmenter          = [segmenter]   # Separate params.
        self.shape_sample       = shape_sample
        self.decoder_autoencode = decoder_autoencode
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_ae        = lambda_x_ae
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_slice       = lambda_slice
    
    def _z_sample(self, batch_size, rng=None):
        if rng is None:
            rng = self.rng
        print('shape sample')
        print(self.shape_sample)
        sample = rng.randn(batch_size, *self.shape_sample).astype(np.float32)
        print('sample')
        print(sample.shape)
        ret = Variable(torch.from_numpy(sample))
        ret = ret.to(torch.cuda.current_device())
        return ret
    
    @autocast_if_needed()
    def forward(self, x_A, x_B, class_A=None, class_B=None, rng=None):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, skip_A = self.encoder(x_A)
        s_B = skip_B = None
        if (   self.lambda_disc
            or self.lambda_x_ae
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
        x_AB = x_AB_residual = x_AA = x_AB_list = x_AA_list = None
        if (self.lambda_seg
         or self.lambda_disc or self.lambda_x_id or self.lambda_z_id):
            info_AB = {'skip_info': skip_A}
            if class_A is not None:
                info_AB['class_info'] = class_A
            print("DECODER RESIDUAL")
            x_AB_residual, skip_AM = self.decoder_residual(s_A, **info_AB)
        # TODO preco toto sa deje?
        print('s_A shape')
        print(s_A.shape)
        print('s_A.size(1)-self.shape_sample[0]')
        print(s_A.size(1)-self.shape_sample[0])
        print("self.shape_sample[0]")
        print(self.shape_sample[0])
        c_A, u_A = torch.split(s_A, [s_A.size(1)-self.shape_sample[0],
                                     self.shape_sample[0]], dim=1)
        print("c_A")
        print(c_A.shape)
        print("u_A")
        print(u_A.shape)

        if self.lambda_disc or self.lambda_x_id or self.lambda_z_id:
            print('DECODER COMMON')
            x_AB, _ = self.decoder_common(c_A, **info_AB)
            x_AA = add(x_AB, x_AB_residual)
            
            # Unpack.
            x_AA, x_AA_list = unpack(x_AA)
            x_AB, x_AB_list = unpack(x_AB)
            x_AB_residual, _= unpack(x_AB_residual)
        
        # B->(B, dA)->A
        x_BA = x_BA_residual = x_BB = z_BA = u_BA = c_B = None
        x_BA_list = x_BB_list = None
        if self.lambda_disc or self.lambda_x_id or self.lambda_z_id:
            info_BA = {'skip_info': skip_B}
            if class_A is not None:
                info_BA['class_info'] = class_B
            u_BA = self._z_sample(batch_size, rng=rng)
            c_B  = s_B[:,:s_B.size(1)-self.shape_sample[0]]
            print('c_B')
            print(c_B.shape)
            print('s_B')
            print(s_B.shape)
            print('s_Bsize(1)-self.shape_sample[0]')
            print(s_B.size(1)-self.shape_sample[0])
            # TODO u_BA je zle, co to vobec je dpc?
            print('u_BA')
            print(u_BA.shape)

            z_BA = torch.cat([c_B, u_BA], dim=1)
            print('z_BA')
            print(z_BA.shape)
            print('DECODER COMMON')
            x_BB, _ = self.decoder_common(c_B, **info_BA)
            print('x_bb shape')
            print(x_BB.shape)
            print('DECODER RESIDUAL')
            x_BA_residual, _ = self.decoder_residual(z_BA, **info_BA)
            print('x_ba shape')
            print(x_BA_residual.shape)
            x_BA = add(x_BB, x_BA_residual)
            
            # Unpack.
            x_BA, x_BA_list = unpack(x_BA)
            x_BB, x_BB_list = unpack(x_BB)
            x_BA_residual, _= unpack(x_BA_residual)
        
        # Optional separate autoencoder.
        x_AA_ae = x_BB_ae = None
        if self.lambda_x_ae and self.decoder_autoencode is not None:
            print('not using')
            x_AA_ae, _ = self.decoder_autoencode(s_A, skip_info=skip_A)
            x_BB_ae, _ = self.decoder_autoencode(s_B, skip_info=skip_B)
        
        # Segment.
        x_AM = None
        if self.lambda_seg:
            if self.segmenter[0] is not None:
                x_AM = self.segmenter[0](s_A, skip_info=skip_AM)
            else:
                # Re-use residual decoder in mode 1.
                info_AM = {'skip_info': skip_AM}
                if class_A is not None:
                    info_AM['class_info'] = class_A
                print("DECODER RESIDUAL")
                x_AM = self.decoder_residual(s_A, **info_AM, mode=1)
                x_AM, _ = unpack(x_AM)
        
        # Reconstruct latent codes.
        s_BA = s_AA = c_AB = c_BB = None
        print('lambda_z_id')
        print(self.lambda_z_id)
        print('lambda_cyc')
        print(self.lambda_cyc)
        if self.lambda_z_id or self.lambda_cyc:
            print('bugged')
            s_BA, skip_BA = self.encoder(x_BA)
            s_AA, _       = self.encoder(x_AA)
            s_AB, _       = self.encoder(x_AB)
            s_BB, _       = self.encoder(x_BB)
            c_AB = s_AB[:,:s_AB.size(1)-self.shape_sample[0]]
            c_BB = s_BB[:,:s_BB.size(1)-self.shape_sample[0]]
            print('bugged end')
        
        # Cycle.
        x_BAB = c_BA = u_BA = None
        if self.lambda_cyc:
            info_BAB = {'skip_info': skip_BA}
            if class_A is not None:
                info_BAB['class_info'] = class_B
            c_BA, u_BA = torch.split(s_BA, [s_BA.size(1)-self.shape_sample[0],
                                            self.shape_sample[0]], dim=1)
            print("DECODER COMMON")
            x_BAB, _ = self.decoder_common(c_BA, **info_BAB)
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
            ('x_AA_ae',       x_AA_ae),
            ('x_BB_ae',       x_BB_ae),
            ))
        hidden = OrderedDict((
            ('s_BA',          s_BA),
            ('s_AA',          s_AA),
            ('c_AB',          c_AB),
            ('c_BB',          c_BB),
            ('z_BA',          z_BA),
            ('s_A',           s_A),
            ('c_A',           c_A),
            ('u_A',           u_A),
            ('c_B',           c_B),
            ('c_BA',          c_BA),
            ('u_BA',          u_BA)
            ))
        intermediates = OrderedDict((
            ('x_AA_list',     x_AA_list),
            ('x_AB_list',     x_AB_list),
            ('x_BB_list',     x_BB_list),
            ('x_BA_list',     x_BA_list),
            ('skip_A',        skip_A),
            ('skip_B',        skip_B)
            ))
        return visible, hidden, intermediates


class _loss_D(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, classifier_A=None, 
                 classifier_B=None, mi_estimator=None, scaler=None,
                 lambda_disc=1, lambda_x_ae=10, lambda_x_id=10, lambda_z_id=1,
                 lambda_f_id=1, lambda_seg=1, lambda_cyc=0, lambda_mi=0,
                 lambda_slice=0, debug_ac_gan=False):
        super(_loss_D, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.lambda_disc        = lambda_disc
        self.lambda_x_ae        = lambda_x_ae
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_slice       = lambda_slice
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_A'    : disc_A,
                    'disc_B'    : disc_B,
                    'class_A'   : classifier_A,
                    'class_B'   : classifier_B,
                    'mi'        : mi_estimator}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_A, x_B, out_BA, out_AB, c_A, u_A, c_BA, u_BA,
                class_A=None, class_B=None):
        # Detach all tensors; updating discriminator, not generator.
        if isinstance(out_BA, list):
            out_BA = [x.detach() for x in out_BA]
        else:
            out_BA = out_BA.detach()
        if isinstance(out_AB, list):
            out_AB = [x.detach() for x in out_AB]
        else:
            out_AB = out_AB.detach()
        c_A = c_A.detach()
        u_A = u_A.detach()
        c_BA = c_BA.detach()
        u_BA = u_BA.detach()
        
        # If outputs are lists, get the last item (image).
        x_BA = out_BA
        x_AB = out_AB
        if not isinstance(x_BA, torch.Tensor):
            x_BA = out_BA[-1]
        if not isinstance(x_AB, torch.Tensor):
            x_AB = out_AB[-1]
        
        # Discriminators.
        kwargs_real = None if class_A is None else {'class_info': class_A}
        kwargs_fake = None if class_B is None else {'class_info': class_B}
        loss_disc = OrderedDict()
        loss_disc_A = self._gan.D(self.net['disc_A'],
                                  fake=out_BA,
                                  real=x_A,
                                  kwargs_real=kwargs_real,
                                  kwargs_fake=kwargs_fake,
                                  scaler=self.scaler)
        loss_disc['A'] = loss_disc_A
        kwargs_real = None if class_A is None else {'class_info': class_B}
        kwargs_fake = None if class_B is None else {'class_info': class_A}
        loss_disc_B = self._gan.D(self.net['disc_B'],
                                  fake=out_AB,
                                  real=x_B,
                                  kwargs_real=kwargs_real,
                                  kwargs_fake=kwargs_fake,
                                  scaler=self.scaler)
        loss_disc['B'] = loss_disc_B
        
        # Slice number classification.
        loss_slice_est = defaultdict(int)
        if self.lambda_slice and class_A is not None:
            loss_slice_est['A'] = _cce(self.net['class_A'](x_A), class_A)
            if self.debug_ac_gan:
                loss_slice_est['BA'] = _cce(self.net['class_A'](x_BA), 
                                            class_A)
        if self.lambda_slice and class_B is not None:
            loss_slice_est['B'] = _cce(self.net['class_B'](x_B), class_B)
            if self.debug_ac_gan:
                loss_slice_est['AB'] = _cce(self.net['class_B'](x_AB),
                                            class_B)
        
        # Mutual information estimate.
        loss_mi_est = defaultdict(int)
        if self.net['mi'] is not None:
            loss_mi_est['A'] = self.net['mi'](c_A, u_A)
            if self.lambda_cyc:
                loss_mi_est['BA'] = self.net['mi'](c_BA, u_BA)
        
        return loss_disc, loss_slice_est, loss_mi_est


class _loss_G(nn.Module):
    def __init__(self, gan_objective, disc_A, disc_B, classifier_A=None, 
                 classifier_B=None, mi_estimator=None, scaler=None,
                 loss_rec=mae, lambda_disc=1, lambda_x_ae=10, lambda_x_id=10,
                 lambda_z_id=1, lambda_f_id=1, lambda_seg=1, lambda_cyc=0,
                 lambda_mi=0, lambda_slice=0, debug_ac_gan=False):
        super(_loss_G, self).__init__()
        self._gan               = gan_objective
        self.scaler             = scaler
        self.loss_rec           = loss_rec
        self.lambda_disc        = lambda_disc
        self.lambda_x_ae        = lambda_x_ae
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_f_id        = lambda_f_id
        self.lambda_seg         = lambda_seg
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_slice       = lambda_slice
        self.debug_ac_gan       = debug_ac_gan
        self.net = {'disc_A'    : disc_A,
                    'disc_B'    : disc_B,
                    'class_A'   : classifier_A,
                    'class_B'   : classifier_B,
                    'mi'        : mi_estimator}  # Separate params.
    
    @autocast_if_needed()
    def forward(self, x_AM, x_A, x_AB, x_AA, x_B, x_BA, x_BB, x_BAB,
                s_BA, s_AA, c_AB, c_BB, z_BA, s_A, c_A, u_A, c_B, c_BA, u_BA,
                x_AA_list, x_AB_list, x_BB_list, x_BA_list, skip_A, skip_B,
                x_AA_ae=None, x_BB_ae=None, class_A=None, class_B=None):
        ("'G' FORWARD FUNCTION")
        # Mutual information loss for generator.
        loss_mi_gen = defaultdict(int)
        if self.net['mi'] is not None and self.lambda_mi:
            loss_mi_gen['A']  = -self.lambda_mi*self.net['mi'](c_A, u_A)
            if self.lambda_cyc:
                loss_mi_gen['BA'] = -self.lambda_mi*self.net['mi'](c_BA, u_BA)
        
        # Slice number classification.
        loss_slice_gen = defaultdict(int)
        if self.lambda_slice and class_A is not None:
            loss_slice_gen['BA'] = _cce(self.net['class_A'](x_BA), class_A)
        if self.lambda_slice and class_B is not None:
            loss_slice_gen['AB'] = _cce(self.net['class_B'](x_AB), class_B)
        
        # Generator loss.
        loss_gen = defaultdict(int)
        if self.lambda_disc:
            kwargs_real = None if class_A is None else {'class_info': class_B}
            kwargs_fake = None if class_B is None else {'class_info': class_A}
            loss_gen['AB'] = self.lambda_disc*self._gan.G(
                                    self.net['disc_B'],
                                    fake=x_AB, real=x_B,
                                    kwargs_real=kwargs_real,
                                    kwargs_fake=kwargs_fake)
            kwargs_real = None if class_A is None else {'class_info': class_A}
            kwargs_fake = None if class_B is None else {'class_info': class_B}
            loss_gen['BA'] = self.lambda_disc*self._gan.G(
                                    self.net['disc_A'],
                                    fake=x_BA, real=x_A,
                                    kwargs_real=kwargs_real,
                                    kwargs_fake=kwargs_fake)
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        if self.lambda_x_id:
            loss_rec['AA'] = self.lambda_x_id*self.loss_rec(x_AA, x_A)
            loss_rec['BB'] = self.lambda_x_id*self.loss_rec(x_BB, x_B)
        if self.lambda_x_ae and x_AA_ae is not None:
            loss_rec['AA_ae'] = self.lambda_x_ae*self.loss_rec(x_AA_ae, x_A)
        if self.lambda_x_ae and x_BB_ae is not None:
            loss_rec['BB_ae'] = self.lambda_x_ae*self.loss_rec(x_BB_ae, x_B)
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
                  +_reduce(loss_mi_gen.values())
                  +_reduce(loss_slice_gen.values()))
        
        # Compile outputs and return.
        losses = OrderedDict((
            ('l_G',           loss_G),
            ('l_gen_AB',      _reduce([loss_gen['AB']])),
            ('l_gen_BA',      _reduce([loss_gen['BA']])),
            ('l_rec',         _reduce([loss_rec['AA'], loss_rec['BB']])),
            ('l_rec_AA',      _reduce([loss_rec['AA']])),
            ('l_rec_BB',      _reduce([loss_rec['BB']])),
            ('l_rec_AA_ae',   _reduce([loss_rec['AA_ae']])),
            ('l_rec_BB_ae',   _reduce([loss_rec['BB_ae']])),
            ('l_rec_c',       _reduce([loss_rec['z_AB'], loss_rec['z_BB']])),
            ('l_rec_s',       _reduce([loss_rec['z_BA'], loss_rec['z_AA']])),
            ('l_rec_z_BA',    _reduce([loss_rec['z_BA']])),
            ('l_rec_z_AB',    _reduce([loss_rec['z_AB']])),
            ('l_rec_z_AA',    _reduce([loss_rec['z_AA']])),
            ('l_rec_z_BB',    _reduce([loss_rec['z_BB']])),
            ('l_mi',          _reduce([loss_mi_gen['A'], loss_mi_gen['BA']])),
            ('l_mi_A',        _reduce([loss_mi_gen['A']])),
            ('l_mi_BA',       _reduce([loss_mi_gen['BA']])),
            ('l_slice',       _reduce([loss_slice_gen['BA'],
                                       loss_slice_gen['AB']])),
            ('l_slice_BA',    _reduce([loss_slice_gen['BA']])),
            ('l_slice_AB',    _reduce([loss_slice_gen['AB']]))
            ))
        return losses
