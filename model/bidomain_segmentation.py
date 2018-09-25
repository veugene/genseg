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


class _gradient_reflow_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_a, input_b, projection_ab):
        ctx.projection_ab = projection_ab
        return input_a
    
    @staticmethod
    def backward(ctx, output_grad):
        grad_a = output_grad
        projection_ab = ctx.projection_ab
        if grad_a.is_cuda:
            projection_ab = projection_ab.cuda()
        with torch.no_grad():
            grad_b = -torch.einsum('bcxy,cn->bnxy', (grad_a, projection_ab))
        return grad_a, grad_b, None


class gradient_reflow(nn.Module):
    def __init__(self, channels_a, channels_b, rng=None):
        super(gradient_reflow, self).__init__()
        self.channels_a = channels_a
        self.channels_b = channels_b
        self.rng = rng if rng else np.random.RandomState()
        projection = self.rng.rand(channels_a,
                                   channels_b).astype(np.float32)+0.1
        self.projection = Variable(torch.from_numpy(projection),
                                   requires_grad=False)
        
    def forward(self, input_a, input_b):
        output = _gradient_reflow_function.apply(input_a, input_b,
                                                 self.projection)
        return output


class segmentation_model(nn.Module):
    def __init__(self, encoder, decoder, disc_A, disc_B, shape_common,
                 shape_unique, mutual_information=None, classifier=None,
                 class_grad_reflow=False, loss_rec=mae, loss_seg=None,
                 lambda_disc=1, lambda_x_id=10, lambda_z_id=1, lambda_const=1,
                 lambda_cyc=0, lambda_mi=1, lambda_seg=1, lambda_class=1,
                 rng=None, debug_no_constant=False, debug_scaling=False,
                 debug_vis_udecode=False):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.encoder          = encoder
        self.decoder          = decoder
        self.disc = {'A' :        disc_A,
                     'B' :        disc_B}   # (separate params)
        self.loss_rec           = loss_rec
        self.loss_seg           = loss_seg if loss_seg else dice_loss()
        self.shape_common       = shape_common
        self.shape_unique       = shape_unique
        self.class_grad_reflow  = class_grad_reflow
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_const       = lambda_const
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_seg         = lambda_seg
        self.lambda_class       = lambda_class
        self.debug_no_constant  = debug_no_constant
        self.debug_scaling      = debug_scaling
        self.debug_vis_udecode  = debug_vis_udecode
        self.is_cuda            = False
        self._shape = {'common': shape_common,
                       'unique': shape_unique}
        
        # Set up mutual information estimator and a classifier.
        mi_estimator = None
        if mutual_information is not None:
            mi_estimator = mine(mutual_information, rng=self.rng)
        self.estimator = {'mi':    mi_estimator,
                          'class': classifier}  # (separate params)
        
        # Random inverse gradient projection from common to sampled features,
        # for use with classifier.
        self.gradient_reflow = gradient_reflow(channels_a=self.shape_common[0],
                                               channels_b=self.shape_unique[0],
                                               rng=self.rng)
    
    def _z_constant(self, batch_size, name='unique'):
        if 0 in self._shape[name]:
            ret = Variable(torch.Tensor([]))
        else:
            ret = Variable(torch.zeros((batch_size,)+self._shape[name],
                                        dtype=torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _z_sample(self, batch_size, name='unique', rng=None):
        if rng is None:
            rng = self.rng
        sample = rng.randn(batch_size, *self._shape[name]).astype(np.float32)
        ret = Variable(torch.from_numpy(sample))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _call_on_all_models(self, fname, *args, **kwargs):
        for key in self.estimator:
            if self.estimator[key] is None: continue
            self.estimator[key] = getattr(self.estimator[key],
                                          fname)(*args, **kwargs)
        for key in self.disc:
            if self.disc[key] is None: continue
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
        
    def encode(self, x):
        z_c, z_u, skip_info = self.encoder(x)
        z = {'common'  : z_c,
             'unique'  : z_u}
        return z, z_c, z_u, skip_info
        
    def decode(self, common, unique, skip_info, scale=None, out_idx=0):
        if self.debug_no_constant or not self.debug_scaling:
            scale = None
        if scale=='common':
            s = (common.size(1)+unique.size(1))/float(max(common.size(1), 1.))
            common = common*s
        if scale=='unique':
            s = (common.size(1)+unique.size(1))/float(max(unique.size(1), 1.))
            unique = unique*s
        out = self.decoder(common, unique, skip_info, transform_index=out_idx)
        return out
    
    def translate_AB(self, x_A, rng=None):
        batch_size = len(x_A)
        s_A, _, _, skip_A = self.encode(x_A)
        z_AB = {'common'  : s_A['common'],
                'unique'  : self._z_constant(batch_size),
                'scale'   : 'common'}
        if self.debug_no_constant:
            z_AB['unique'] = self._z_sample(batch_size, rng=rng)
        x_AB = self.decode(**z_AB, skip_info=skip_A)
        return x_AB
    
    def translate_BA(self, x_B, rng=None):
        batch_size = len(x_B)
        s_B, _, _, skip_B = self.encode(x_B)
        z_BA = {'common'  : s_B['common'],
                'unique'  : self._z_sample(batch_size, rng=rng),
                'scale'   : None}
        x_BA = self.decode(**z_BA, skip_info=skip_B)
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_A)
        s_A, _, skip_A = self.encode(x_A)
        z_AM = {'common'  : self._z_constant(batch_size, 'common'),
                'unique'  : s_A['unique'],
                'scale'   : 'unique'}
        x_AM = self.decode(**z_AM, skip_info=skip_A, out_idx=1)
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
        s_A, c_A, u_A, skip_A = self.encode(x_A)
        only_seg = True
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id
            or self.lambda_const
            or self.lambda_cyc
            or self.lambda_mi):
                s_B, c_B, u_B, skip_B = self.encode(x_B)
                only_seg = False
        
        # Reconstruct inputs.
        x_AA = x_BB = None
        if self.lambda_x_id:
            z_AA = {'common'  : s_A['common'],
                    'unique'  : s_A['unique'],
                    'scale'   : None}
            z_BB = {'common'  : s_B['common'],
                    'unique'  : self._z_constant(batch_size),
                    'scale'   : 'common'}
            if self.debug_no_constant:
                z_BB['unique'] = s_B['unique']
            x_AA = self.decode(**z_AA, skip_info=skip_A)
            x_BB = self.decode(**z_BB, skip_info=skip_B)
        
        # Translate.
        x_AB = x_BA = None
        if self.lambda_disc or self.lambda_z_id:
            z_AB = {'common'  : s_A['common'],
                    'unique'  : self._z_constant(batch_size),
                    'scale'   : 'common'}
            if self.debug_no_constant:
                z_AB['unique'] = self._z_sample(batch_size, rng=rng)
            z_BA = {'common'  : s_B['common'],
                    'unique'  : self._z_sample(batch_size, rng=rng),
                    'scale'   : None}
            x_AB = self.decode(**z_AB, skip_info=skip_A)
            x_BA = self.decode(**z_BA, skip_info=skip_B)
        
        # Reconstruct latent codes.
        if self.lambda_z_id or self.lambda_cyc:
            s_AB, a_AB, b_AB, skip_AB = self.encode(x_AB)
            s_BA, a_BA, b_BA, skip_BA = self.encode(x_BA)
        
        # Cycle.
        x_ABA = x_BAB = None
        if self.lambda_cyc:
            z_ABA = {'common'  : s_AB['common'],
                     'unique'  : s_A['unique'],
                     'scale'   : None}
            z_BAB = {'common'  : s_BA['common'],
                     'unique'  : s_B['unique'],
                     'scale'   : None}
            x_ABA = self.decode(**z_ABA, skip_info=skip_AB)
            x_BAB = self.decode(**z_BAB, skip_info=skip_BA)
        
        # Segment.
        x_AM = None
        if self.lambda_seg and mask is not None and len(mask)>0:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            z_AM = {'common'  : self._z_constant(num_masks, 'common'),
                    'unique'  : s_A['unique'][mask_indices],
                    'scale'   : 'unique'}
            skip_A_filtered = [s[mask_indices] for s in skip_A]
            x_AM = self.decode(**z_AM, skip_info=skip_A_filtered, out_idx=1)
        
        # Debug decode of u alone (non-seg output).
        x_AU = None
        if self.debug_vis_udecode:
            z_AU = {'common'  : self._z_constant(batch_size, 'common'),
                    'unique'  : s_A['unique'],
                    'scale'   : 'unique'}
            with torch.no_grad():
                x_AU = self.decode(**z_AU, skip_info=skip_A)
        
        # Estimator losses
        #
        # Classifier.
        loss_class_est = 0
        probabilities = defaultdict(int)
        if self.estimator['class'] is not None and not only_seg:
            def classify(x):
                logit = self.estimator['class'](x.view(batch_size, -1))
                return torch.sigmoid(logit)
            probabilities['A'] = classify(c_A.detach())
            probabilities['B'] = classify(c_B.detach())
            loss_class_est = ( bce(probabilities['A'], 1)
                              +bce(probabilities['B'], 0))
            if optimizer is not None:
                loss_class_est.mean().backward()
        # Mutual information.
        loss_mi_est = defaultdict(int)
        if self.estimator['mi'] is not None:
            loss_mi_est['A']  = self.estimator['mi'](c_A.detach(),
                                                     u_A.detach())
            if self.lambda_z_id or self.lambda_cyc:
                loss_mi_est['BA'] = self.estimator['mi'](a_BA.detach(),
                                                         b_BA.detach())
            if optimizer is not None:
                loss_mi_est['A'].mean().backward()
                if self.lambda_z_id or self.lambda_cyc:
                    loss_mi_est['BA'].mean().backward()        
                optimizer['E'].step()
        
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
        
        # Reconstruction loss.
        loss_rec = defaultdict(int)
        dist = self.loss_rec
        if self.lambda_x_id:
            loss_rec['AA'] = self.lambda_x_id*dist(x_AA, x_A)
            loss_rec['BB'] = self.lambda_x_id*dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_rec['zc_AB'] = self.lambda_z_id*dist(s_AB['common'],
                                                      z_AB['common'])
            loss_rec['zu_AB'] = self.lambda_z_id*dist(s_AB['unique'],
                                                      z_AB['unique'])
            loss_rec['zc_BA'] = self.lambda_z_id*dist(s_BA['common'],
                                                      z_BA['common'])
            loss_rec['zu_BA'] = self.lambda_z_id*dist(s_BA['unique'],
                                                      z_BA['unique'])
        
        # Latent factor classifier loss for generator.
        loss_class_gen = 0
        if self.lambda_class and self.estimator['class'] is not None \
                             and not only_seg:
            def classify(x):
                logit = self.estimator['class'](x.view(batch_size, -1))
                return torch.sigmoid(logit)
            if self.class_grad_reflow:
                # Features removed in 'c' pushed into 'u'.
                c_A_ = self.gradient_reflow(c_A, u_A)
                c_B_ = self.gradient_reflow(c_B, u_B)
            else:
                c_A_ = c_A
                c_B_ = c_A
            loss_class_gen = ( bce(classify(c_A_), 0.5)
                              +bce(classify(c_B_), 0.5))
            loss_class_gen = self.lambda_class*loss_class_gen
        
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
        if self.lambda_mi and self.estimator['mi'] is not None:
            loss_mi_gen['A']  = -self.lambda_mi*self.estimator['mi'](c_A,u_A)
            loss_mi_gen['BA'] = -self.lambda_mi*self.estimator['mi'](a_BA,b_BA)
        
        # Segmentation loss.
        loss_seg = 0
        if self.lambda_seg and mask is not None and len(mask)>0:
            loss_seg = self.lambda_seg*self.loss_seg(x_AM, mask)
        
        # All generator losses combined.
        loss_G = ( _reduce(loss_gen.values())
                  +_reduce(loss_rec.values())
                  +_reduce(loss_cyc.values())
                  +_reduce(loss_mi_gen.values())
                  +_reduce([loss_const_B])
                  +_reduce([loss_class_gen])
                  +_reduce([loss_seg]))
        
        # Compute generator gradients.
        if optimizer is not None and isinstance(loss_G, torch.Tensor):
            loss_G.mean().backward()
            optimizer['G'].step()
        
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
            ('l_rec_zu_AB', _reduce([loss_rec['zu_AB']])),
            ('l_rec_z_AB',  _reduce(_cat([loss_rec['zc_AB'],
                                          loss_rec['zu_AB']], dim=1))),
            ('l_rec_zc_BA', _reduce([loss_rec['zc_BA']])),
            ('l_rec_zu_BA', _reduce([loss_rec['zu_BA']])),
            ('l_rec_z_BA',  _reduce(_cat([loss_rec['zc_BA'],
                                          loss_rec['zu_BA']], dim=1))),
            ('l_rec_z',     _reduce([ _cat([loss_rec['zc_AB'],
                                            loss_rec['zu_AB']], dim=1)
                                     +_cat([loss_rec['zc_BA'],
                                            loss_rec['zu_BA']], dim=1)])),
            ('l_class',     _reduce([loss_class_gen])),
            ('l_const_B',   _reduce([loss_const_B])),
            ('l_cyc_ABA',   _reduce([loss_cyc['ABA']])),
            ('l_cyc_BAB',   _reduce([loss_cyc['BAB']])),
            ('l_cyc',       _reduce([loss_cyc['ABA']+loss_cyc['BAB']])),
            ('l_seg',       _reduce([loss_seg])),
            ('l_mi_A',      loss_mi_est['A']),
            ('l_mi_BA',     loss_mi_est['BA']),
            ('l_mi',        loss_mi_est['A']+loss_mi_est['BA']),
            ('out_AA',      x_AA),
            ('out_BB',      x_BB),
            ('out_AB',      x_AB),
            ('out_BA',      x_BA),
            ('out_ABA',     x_ABA),
            ('out_BAB',     x_BAB),
            ('out_seg',     x_AM),
            ('out_u_dec',   x_AU),
            ('mask',        mask),
            ('prob_A',      _reduce([probabilities['A']])),
            ('prob_B',      _reduce([1-probabilities['B']]))))
        return outputs
