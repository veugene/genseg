import torch
from torch import nn
import torch.nn.functional as F


def dist_ratio_mse_abs(prediction, target, eps=1e-7, reduce=False):
    loss = mse(prediction, target, reduce=False)/(mae(prediction, target)+eps)
    if reduce:
        loss = torch.mean(loss)
    return loss
    

def bce(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.to(prediction.device)
    return F.binary_cross_entropy(prediction, target, reduce=reduce)


def mse(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.to(prediction.device)
    return F.mse_loss(prediction, target, reduce=reduce)


def mae(prediction, target, reduce=False):
    loss = torch.abs(prediction-target)
    if reduce:
        loss = torch.mean(loss)
    return loss 


class gan_objective(object):
    def __init__(self, objective, relativistic=False,
                 grad_penalty_real=None, grad_penalty_fake=None,
                 grad_penalty_mean=0):
        # jenson_shannon
        # least_squares
        # hinge
        # wasserstein
        self.objective = objective
        self.relativistic = relativistic
        self.grad_penalty_real = grad_penalty_real
        self.grad_penalty_fake = grad_penalty_fake
        self.grad_penalty_mean = grad_penalty_mean
        if   objective=='jenson_shannon':
            self._D1 = lambda x : bce(torch.sigmoid(x), 1)
            self._D0 = lambda x : bce(torch.sigmoid(x), 0)
            self._G  = lambda x : bce(torch.sigmoid(x), 1)
        elif objective=='least_squares':
            self._D1 = lambda x : mse(x, 1)
            self._D0 = lambda x : mse(x, 0)
            self._G  = lambda x : mse(x, 1)
        elif objective=='hinge':
            self._D1 = lambda x : F.relu(1.-x)
            self._D0 = lambda x : F.relu(1.+x)
            self._G  = lambda x : -x
        elif objective=='wasserstein':
            self._D1 = lambda x : -x
            self._D0 = lambda x : x
            self._G  = lambda x : -x
        else:
            raise ValueError("Unknown objective: {}".format(objective))
    
    def G(self, disc, fake, real=None, kwargs_fake=None, kwargs_real=None):
        if kwargs_fake is None: kwargs_fake = {}
        if kwargs_real is None: kwargs_real = {}
        if self.relativistic:
            return self._foreach(lambda x: self._G(x[0]-x[1]),
                                 [disc(fake, **kwargs_fake),
                                  disc(real, **kwargs_real)])
        return self._foreach(self._G, disc(fake, **kwargs_fake))
    
    def D(self, disc, fake, real, kwargs_fake=None, kwargs_real=None):
        if kwargs_fake is None: kwargs_fake = {}
        if kwargs_real is None: kwargs_real = {}
        if self.relativistic:
            loss_real = self._D_relativistic(real, fake, disc, self._D1,
                                             self.grad_penalty_real,
                                             self.grad_penalty_fake,
                                             kwargs_a=kwargs_real,
                                             kwargs_b=kwargs_fake)
            loss_fake = self._D_relativistic(fake, real, disc, self._D0,
                                             self.grad_penalty_fake,
                                             self.grad_penalty_real,
                                             kwargs_a=kwargs_fake,
                                             kwargs_b=kwargs_real)
        else:
            loss_real = self._D(real, disc, self._D1, self.grad_penalty_real,
                                kwargs_disc=kwargs_real)
            loss_fake = self._D(fake, disc, self._D0, self.grad_penalty_fake,
                                kwargs_disc=kwargs_fake)
        return loss_real+loss_fake
    
    def _D(self, x, disc, objective, grad_penalty, kwargs_disc=None):
        if kwargs_disc is None: kwargs_disc = {}
        disc_out = disc(x, **kwargs_disc)
        loss = self._foreach(objective, disc_out)
        if grad_penalty is not None:
            if isinstance(disc_out, torch.Tensor):
                disc_out = [disc_out]
            disc_out = sum([o.sum() for o in disc_out])
            grad = torch.autograd.grad(disc_out,
                                       x,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            norm2 = (grad.view(grad.size()[0],-1)**2).sum(-1)
            if self.grad_penalty_mean:
                penalty = (torch.sqrt(norm2)-self.grad_penalty_mean)**2
            else:
                penalty = norm2
            loss = loss+grad_penalty*torch.mean(penalty)
        return loss
    
    def _D_relativistic(self, a, b, disc, objective,
                        grad_penalty_a, grad_penalty_b,
                        kwargs_a=None, kwargs_b=None):
        if kwargs_a is None: kwargs_a = {}
        if kwargs_b is None: kwargs_b = {}
        if torch.is_grad_enabled():
            a.requires_grad = True
            b.requires_grad = True
        disc_a = disc(a, **kwargs_a)
        disc_b = disc(b, **kwargs_b)
        loss = self._foreach(lambda x: objective(x[0]-x[1]), [disc_a, disc_b])
        if torch.is_grad_enabled():
            if grad_penalty_a is not None or grad_penalty_b is not None:
                if isinstance(disc_a, torch.Tensor):
                    disc_a = [disc_a]
                if isinstance(disc_b, torch.Tensor):
                    disc_b = [disc_b]
                disc_out = sum([o.sum() for o in disc_a+disc_b])
                grad_a, grad_b = torch.autograd.grad(disc_out,
                                                     (a, b),
                                                     retain_graph=True,
                                                     create_graph=True,
                                                     only_inputs=True)
                norm2_a = (grad_a.view(grad_a.size()[0],-1)**2).sum(-1)
                norm2_b = (grad_b.view(grad_b.size()[0],-1)**2).sum(-1)
                if self.grad_penalty_mean:
                    penalty_a = (torch.sqrt(norm2_a)-self.grad_penalty_mean)**2
                    penalty_b = (torch.sqrt(norm2_b)-self.grad_penalty_mean)**2
                else:
                    penalty_a = norm2_a
                    penalty_b = norm2_b
                if grad_penalty_a is not None:
                    loss = loss+grad_penalty_a*torch.mean(penalty_a)
                if grad_penalty_b is not None:
                    loss = loss+grad_penalty_b*torch.mean(penalty_b)
        return loss
    
    def _foreach(self, func, x):
        # If x is a list, process every element (and reduce to batch dim).
        # Each tensor is reduced by `mean` and reduced tensors are averaged
        # together.
        # (For multi-scale discriminators. Each scale is given equal weight.)
        if not isinstance(x, torch.Tensor):
            return sum([self._foreach(func, elem) for elem in x])/float(len(x))
        out = func(x)
        if out.dim()<=1:
            return out
        return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.


class dice_loss(torch.nn.Module):
    '''
    Dice loss.
    
    Expects integer or one-hot class labeling in y_true.
    Expects outputs in range [0, 1] in y_pred.
    
    Computes the soft dice loss considering all classes in target_class as one
    aggregate target class and ignoring all elements with ground truth classes
    in mask_class.
    
    target_class : integer or list, integer class(es) to use from target.
    prediction_index : integer or list, channel index corresponding to each
        class.
    mask_class : integer or list, class(es) specifying points at which
        not to compute a loss.
    '''
    def __init__(self, target_class=1, prediction_index=0, mask_class=None):
        super(dice_loss, self).__init__()
        if not hasattr(target_class, '__len__'):
            target_class = [target_class]
        if not hasattr(prediction_index, '__len__'):
            prediction_index = [prediction_index]
        if mask_class is not None and not hasattr(mask_class, '__len__'):
            mask_class = [mask_class]
        self.target_class = target_class
        self.prediction_index = prediction_index
        self.mask_class = mask_class
        self.smooth = 1
            
    def forward(self, y_pred, y_true):
        # Targer variable must not require a gradient.
        assert(not y_true.requires_grad)
        
        # Index into y_pred.
        y_pred = sum([y_pred[:,i:i+1] for i in self.prediction_index])
        #if not y_pred.is_contiguous:
            #y_pred.contiguous()
    
        # If needed, change ground truth from categorical to integer format.
        if y_true.ndimension() > y_pred.ndimension():
            y_true = torch.max(y_true, dim=1)[1]   # argmax
            
        # Flatten all inputs.
        y_true_f = y_true.view(-1).int()
        y_pred_f = y_pred.view(-1)
        
        # Aggregate target classes, mask out classes in mask_class.
        y_target = sum([y_true_f==t for t in self.target_class]).float()
        if self.mask_class is not None:
            mask_out = sum([y_true_f==t for t in self.mask_class])
            idxs = (mask_out==0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        
        # Compute dice value.
        intersection = torch.sum(y_target * y_pred_f)
        dice_val = -(2.*intersection+self.smooth) / \
                    (torch.sum(y_target)+torch.sum(y_pred_f)+self.smooth)
                    
        return dice_val
