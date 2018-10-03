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
            self._D1 = lambda x : -x
            self._D0 = lambda x : x
            self._G  = lambda x : -x
        else:
            raise ValueError("Unknown objective: {}".format(objective))
    
    def G(self, disc, fake, real=None):
        if self.relativistic:
            return self._foreach(lambda x: self._G(x[0]-x[1]),
                                 [disc(fake), disc(real)])
        return self._foreach(self._G, disc(fake))
    
    def D(self, disc, fake, real):
        if self.relativistic:
            loss_real = self._D_relativistic(real, fake, disc, self._D1,
                                             self.grad_penalty_real,
                                             self.grad_penalty_fake)
            loss_fake = self._D_relativistic(fake, real, disc, self._D0,
                                             self.grad_penalty_fake,
                                             self.grad_penalty_real)
        else:
            loss_real = self._D(real, disc, self._D1, self.grad_penalty_real)
            loss_fake = self._D(fake, disc, self._D0, self.grad_penalty_fake)
        return loss_real+loss_fake
    
    def _D(self, x, disc, objective, grad_penalty):
        x.requires_grad = True
        disc_out = disc(x)
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
                        grad_penalty_a, grad_penalty_b):
        a.requires_grad = True
        b.requires_grad = True
        disc_a = disc(a)
        disc_b = disc(b)
        loss = self._foreach(lambda x: objective(x[0]-x[1]), [disc_a, disc_b])
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
        # (For multi-scale discriminators).
        if not isinstance(x, torch.Tensor):
            return sum([self._foreach(func, elem) for elem in x])
        out = func(x)
        if out.dim()<=1:
            return out
        return out.view(out.size(0), -1).mean(1)    # Reduce to batch dim.