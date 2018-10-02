import torch
from torch import nn


def dist_ratio_mse_abs(x, y, eps=1e-7):
    return torch.mean((x-y)**2) / (torch.mean(torch.abs(x-y))+eps)
    

def bce(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    return nn.BCELoss(reduce=reduce)(prediction, target)


def mse(prediction, target, reduce=False):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
    return nn.MSELoss(reduce=reduce)(prediction, target)


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
        if relativistic:
            raise NotImplementedError("relativistic=True")
        self.grad_penalty_real = grad_penalty_real
        self.grad_penalty_fake = grad_penalty_fake
        self.grad_penalty_mean = grad_penalty_mean
        if   objective=='jenson_shannon':
            self._D1 = bce(torch.sigmoid(x), 1)
            self._D0 = bce(torch.sigmoid(x), 0)
            self._G  = bce(torch.sigmoid(x), 1)
        elif objective=='least_squares':
            self._D1 = mse(x, 1)
            self._D0 = mse(x, 0)
            self._G  = mse(x, 1)
        elif objective=='hinge':
            self._D1 = lambda x : nn.ReLU()(1.-x)
            self._D0 = lambda x : nn.ReLU()(1.+x)
            self._G  = lambda x : -x
        elif objective=='wasserstein':
            raise NotImplementedError("objective=wasserstein")
        else:
            raise ValueError("Unknown objective: {}".format(objective))
    
    def G(self, disc, fake, real=None):
        return self._foreach(self._G, disc(fake))
    
    def D(self, disc, fake, real):
        loss_real = self._D(real, disc, self._D1, self.grad_penalty_real)
        loss_fake = self._D(fake, disc, self._D0, self.grad_penalty_fake)
        return loss_real+loss_fake
    
    def _D(self, x, disc, objective, grad_penalty_weight):
        if grad_penalty_weight is None:
            return self._foreach(objective, disc(x))
        # Gradient penalty for each item in disc output.
        x.requires_grad = True
        def objective_gp(disc_out):
            grad = torch.autograd.grad(disc_out.sum(),
                                       x,
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]
            norm2 = (grad.view(grad.size()[0],-1)**2).sum(-1)
            norm2 = torch.mean(norm2)
            if self.grad_penalty_mean:
                penalty = (torch.sqrt(norm2)-self.grad_penalty_mean)**2
            else:
                penalty = norm2
            loss = objective(x)+grad_penalty_weight*penalty
            return loss
        return self._foreach(objective_gp, disc(x))
    
    def _foreach(self, func, x):
        # If x is a list, process every element (and reduce to batch dim).
        # (For multi-scale discriminators).
        if not isinstance(x, torch.Tensor):
            return sum([self._foreach(func, elem) for elem in x])
        out = func(x)
        if out.dim()<=1:
            return out
        return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.