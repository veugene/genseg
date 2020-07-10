# ============================================================
#
#  SSL Mean Teacher
#  https://arxiv.org/pdf/1703.01780.pdf
#
# ============================================================

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from fcn_maker.loss import dice_loss
from .common.losses import mse


# TODO: necessary? (Yes if any part of the model is not to be updated during
# unsupervised training.)
def clear_grad(optimizer):
    # Sets `grad` to None instead of zeroing it.
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad = None


def _reduce(loss):
    def _mean(x):
        if not isinstance(x, torch.Tensor) or x.dim()<=1:
            return x
        else:
            return x.view(x.size(0), -1).mean(1)
    if not hasattr(loss, '__len__'): loss = [loss]
    if len(loss)==0: return 0
    return sum([_mean(v) for v in loss])


class segmentation_model(nn.Module):
    def __init__(self, student, teacher, loss_seg=None, lambda_con=10.,
                 alpha_max=0.99):
        super(segmentation_model, self).__init__()
        self.student    = student
        self.teacher    = teacher
        self.loss_seg   = loss_seg if loss_seg else dice_loss()
        self.lambda_con = lambda_con    # consistency weight
        self.alpha_max  = alpha_max     # for exponential moving average
        self.is_cuda    = False
        self._iteration = nn.Parameter(torch.zeros(1), requires_grad=False)

    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        super(segmentation_model, self).cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(segmentation_model, self).cpu(*args, **kwargs)

    def forward(self, x_A, x_B=None, mask=None, optimizer=None, **kwargs):
        compute_grad = True if optimizer is not None else False
        if compute_grad:
            if isinstance(optimizer, dict):
                assert len(optimizer)==1
                optimizer = list(optimizer.values())[0]
            optimizer.zero_grad()
            self._iteration += 1
        x = x_A
        if x_B is not None:
            x = torch.cat([x_A, x_B], dim=0)
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x, mask, optimizer=optimizer)

    def _evaluate(self, x, mask=None, optimizer=None):
        loss_seg = 0
        loss_con = 0
        
        def _mean(x):
            if not isinstance(x, torch.Tensor) or x.dim()<=1:
                return x
            else:
                return x.view(x.size(0), -1).mean(1)
        
        # Predict outputs by student network.
        
        
        # Prepare a mask Tensor without None entries.
        mask_packed = None
        mask_indices = []
        if mask is not None:
            mask_indices = [i for i, m in enumerate(mask) if m is not None]
            mask_packed = np.array([mask[i] for i in mask_indices])
            mask_packed = Variable(torch.from_numpy(mask_packed))
            mask_packed = mask_packed.cuda()
        
        # Separate out unsupervised inputs.
        no_mask_indices = [i for i, m in enumerate(mask) if m is None]
        x_u = x[no_mask_indices]
        
        # Segment.
        x_AM_student = self.student(x)
        x_AM_teacher = self.teacher(x)
        
        # Student segmentation loss.
        x_AM_sup_student = None
        x_AM_sup_teacher = None
        if len(mask_indices):
            x_AM_sup_student = x_AM_student[mask_indices]
            x_AM_sup_teacher = x_AM_teacher[mask_indices]
            loss_seg = _mean(self.loss_seg(x_AM_sup_student, mask_packed))
        
        # Match the teacher (unsupervised).
        x_AM_unsup_student = None
        x_AM_unsup_teacher = None
        if self.lambda_con and len(mask_indices) < len(mask):
            x_AM_unsup_student = x_AM_student[no_mask_indices]
            x_AM_unsup_teacher = x_AM_teacher[no_mask_indices]
            loss_con = _mean(F.mse_loss(x_AM_unsup_student,
                                        x_AM_unsup_teacher))
        
        # Loss. Compute gradients, if requested.
        loss = loss_seg + self.lambda_con*loss_con
        if optimizer is not None:
            loss.mean().backward()
            optimizer.step()
        
        # Update teacher's parameters as exponential moving average of
        # student's parameters.
        alpha = min(1 - 1./(self._iteration+1), self.alpha_max)
        for ema_param, param in zip(self.teacher.parameters(),
                                    self.student.parameters()):
            ema_param.data.mul_(self.alpha_max).add_(1-self.alpha_max,
                                                     param.data)
        
        # Compile outputs and return.
        outputs = OrderedDict((
            ('l_all',   loss),
            ('l_seg',   loss_seg),
            ('l_con',   loss_con),
            ('x_AM',                x_AM_sup_student),
            ('x_AM_sup_teacher',    x_AM_sup_teacher),
            ('x_AM_unsup_student',  x_AM_unsup_student),
            ('x_AM_unsup_teacher',  x_AM_unsup_teacher),
            ('x_M',     mask_packed)))
        return outputs