import numpy as np
import torch
from torch.autograd import Variable


class mine(object):
    """
    MINE: Mutual information neural estimator.
    https://arxiv.org/abs/1801.04062
    
    TODO: reduce bias in MINE gradient estimation.
    """
    def __init__(self, estimation_network, rng=None):
        self.estimation_network = estimation_network
        self.rng = rng if rng else np.random.RandomState()
        
    def evaluate(self, x, z, z_marginal=None):
        if z_marginal is None:
            permutation = self.rng.permutation(len(z))
            z_marginal = z[permutation]
        joint = self.estimation_network(x, z)
        marginal = self.estimation_network(x, z_marginal)
        lower_bound = (  torch.mean(joint)
                       - torch.log(torch.mean(torch.exp(marginal))))
        return -lower_bound
    
    
def mse(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction)*target
        if prediction.is_cuda:
            target = target.cuda()
        target = Variable(target)
    return torch.nn.MSELoss()(prediction, target)
    
    
class segmentation_model(torch.nn.Module):
    def __init__(self, f_factor, f_common, f_residual, f_unique,
                 g_common, g_residual, g_unique, g_output,
                 disc_A, disc_B, mutual_information, loss_segmentation,
                 z_size=(50,), z_constant=0, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_const=1, lambda_cyc=0, lambda_mi=1,
                 lambda_seg=1, rng=None):
        super(segmentation_model, self).__init__()
        self.rng = rng if rng else np.random.RandomState()
        self.f_factor           = f_factor
        self.f_common           = f_common
        self.f_residual         = f_residual
        self.f_unique           = f_unique
        self.g_common           = g_common
        self.g_residual         = g_residual
        self.g_unique           = g_unique
        self.g_output           = g_output
        self.disc_A             = disc_A
        self.disc_B             = disc_B
        self.mutual_information = mutual_information
        self.mi_estimator       = mine(mutual_information, rng=self.rng)
        self.loss_segmentation  = loss_segmentation
        self.z_size             = z_size
        self.z_constant         = z_constant
        self.lambda_disc        = lambda_disc
        self.lambda_x_id        = lambda_x_id
        self.lambda_z_id        = lambda_z_id
        self.lambda_const       = lambda_const
        self.lambda_cyc         = lambda_cyc
        self.lambda_mi          = lambda_mi
        self.lambda_seg         = lambda_seg
        self.is_cuda            = False
        
    def _z_constant(self, batch_size):
        ret = Variable(torch.zeros((batch_size,)+self.z_size,
                                   dtype=torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def _z_sample(self, batch_size):
        ret = Variable(torch.randn((batch_size,)
                                    +self.z_size).type(torch.float32))
        if self.is_cuda:
            ret = ret.cuda()
        return ret
    
    def cuda(self, *args, **kwargs):
        self.is_cuda = True
        super(segmentation_model, self).cuda(*args, **kwargs)
        
    def cpu(self, *args, **kwargs):
        self.is_cuda = False
        super(segmentation_model, self).cpu(*args, **kwargs)
        
    def encode(self, x):
        z_a, z_b = self.f_factor(x)
        z_common = self.f_common(z_b)
        z_residual = self.f_residual(z_b)
        z_unique = self.f_unique(z_a)
        z = {'common'  : z_common,
             'residual': z_residual,
             'unique'  : z_unique}
        return z, z_a, z_b
        
    def decode(self, common, residual, unique):
        out = self.g_output(self.g_common(common),
                            self.g_residual(residual),
                            self.g_unique(unique))
        out = torch.sigmoid(out)
        return out
    
    def translate_AB(self, x_A):
        batch_size = len(x_A)
        s_A, _, _ = self.encode(x_A)
        z_A = {'common'  : s_A['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_constant(batch_size)}
        x_AB = self.decode(**z_A)
        return x_AB
    
    def translate_BA(self, x_B):
        batch_size = len(x_B)
        s_B, _, _ = self.encode(x_B)
        z_B = {'common'  : s_B['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_sample(batch_size)}
        x_BA = self.decode(**z_B)
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_A)
        s_A, _, _ = self.encode(x_A)
        z_AM = {'common'  : self._z_constant(batch_size),
                'residual': self._z_constant(batch_size),
                'unique'  : s_A['unique']}
        x_AM = self.decode(**z_AM)
        return x_AM
    
    def evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                 compute_grad=False):
        with torch.set_grad_enabled(compute_grad):
            return self._evaluate(x_A, x_B, mask, mask_indices, compute_grad)
    
    def _evaluate(self, x_A, x_B, mask=None, mask_indices=None,
                  compute_grad=False):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, a_A, b_A = self.encode(x_A)
        if (   self.lambda_disc
            or self.lambda_x_id
            or self.lambda_z_id
            or self.lambda_z_const
            or self.lambda_cyc
            or self.lambda_mi):
                s_B, a_B, b_B = self.encode(x_B)
        
        # Reconstruct inputs.
        if self.lambda_x_id:
            z_AA = {'common'  : s_A['common'],
                    'residual': s_A['residual'],
                    'unique'  : s_A['unique']}
            z_BB = {'common'  : s_B['common'],
                    'residual': s_B['residual'],
                    'unique'  : self._z_constant(batch_size)}
            x_AA = self.decode(**z_AA)
            x_BB = self.decode(**z_BB)
        
        # Translate.
        x_AB = x_BA = None
        if self.lambda_disc or self.lambda_z_id:
            z_AB = {'common'  : s_A['common'],
                    'residual': self._z_sample(batch_size),
                    'unique'  : self._z_constant(batch_size)}
            z_BA = {'common'  : s_B['common'],
                    'residual': self._z_sample(batch_size),
                    'unique'  : self._z_sample(batch_size)}
            x_AB = self.decode(**z_AB)
            x_BA = self.decode(**z_BA)
        
        # Reconstruct latent codes.
        if self.lambda_z_id:
            s_AB, a_AB, b_AB = self.encode(x_AB)
            s_BA, a_BA, b_BA = self.encode(x_BA)
        
        # Cycle.
        x_ABA = x_BAB = None
        if self.lambda_cyc:
            z_ABA = {'common'  : s_AB['common'],
                     'residual': s_A['residual'],
                     'unique'  : s_A['unique']}
            z_BAB = {'common'  : s_BA['common'],
                     'residual': s_B['residual'],
                     'unique'  : s_B['unique']}
            x_ABA = self.decode(**z_ABA)
            x_BAB = self.decode(**z_BAB)
            
        # Segment.
        x_AM = None
        if self.lambda_seg and mask is not None:
            if mask_indices is None:
                mask_indices = list(range(len(mask)))
            num_masks = len(mask_indices)
            z_AM = {'common'  : self._z_constant(num_masks),
                    'residual': self._z_constant(num_masks),
                    'unique'  : s_A['unique'][mask_indices]}
            x_AM = self.decode(**z_AM)
            if self.lambda_z_id:
                s_AM, a_AM, b_AM = self.encode(x_AM)
        
        # Generator losses.
        loss_G = 0
        dist = torch.nn.L1Loss()
        if self.lambda_disc:
            loss_G += self.lambda_disc * mse(self.disc_B(x_AB), 1)
            loss_G += self.lambda_disc * mse(self.disc_A(x_BA), 1)
        if self.lambda_x_id:
            loss_G += self.lambda_x_id * dist(x_AA, x_A)
            loss_G += self.lambda_x_id * dist(x_BB, x_B)
        if self.lambda_z_id:
            loss_G += self.lambda_z_id * dist(s_AB['common'],
                                              z_AB['common'].detach())
            loss_G += self.lambda_z_id * dist(s_AB['residual'],
                                              z_AB['residual'])   # detached
            loss_G += self.lambda_z_id * dist(s_AB['unique'],
                                              z_AB['unique'])     # detached
            loss_G += self.lambda_z_id * dist(s_BA['common'],
                                              z_BA['common'].detach())
            loss_G += self.lambda_z_id * dist(s_BA['residual'],
                                              z_BA['residual'])   # detached
            loss_G += self.lambda_z_id * dist(s_BA['unique'],
                                              z_BA['unique'])     # detached
        if self.lambda_const:
            loss_G += self.lambda_const * dist(s_B['unique'],
                                               self._z_constant(batch_size))
        if self.lambda_cyc:
            loss_G += self.lambda_cyc * dist(x_ABA, x_A)
            loss_G += self.lambda_cyc * dist(x_BAB, x_B)
        if self.lambda_mi:
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_A, b_A)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_B, b_B)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_AB, b_AB)
            loss_G += self.lambda_mi * self.mi_estimator.evaluate(a_BA, b_BA)
        
        # Segment.
        loss_segmentation = 0
        if self.lambda_seg and mask is not None:
            loss_segmentation = self.loss_segmentation(x_AM,
                                                       mask[mask_indices])
            loss_G += self.lambda_seg * loss_segmentation
            if self.lambda_z_id:
                loss_G += (  self.lambda_z_id
                           * dist(s_AM['unique'], z_AM['unique'].detach()))
        
        # Compute generator gradients.
        if compute_grad:
            loss_G.backward()
        if compute_grad and self.lambda_disc:
            self.disc_A.zero_grad()
            self.disc_B.zero_grad()
        
        # Discriminator losses.
        loss_disc_A = loss_disc_B = 0
        if self.lambda_disc:
            loss_disc_A = (  mse(self.disc_A(x_A), 1)
                           + mse(self.disc_A(x_BA.detach()), 0))
            loss_disc_B = (  mse(self.disc_B(x_B), 1)
                           + mse(self.disc_B(x_AB.detach()), 0))
            loss_G += self.lambda_disc * (loss_disc_A+loss_disc_B)
        if self.lambda_disc and compute_grad:
            loss_disc_A.backward()
            loss_disc_B.backward()
        
        # Compile outputs and return.
        losses  = {'seg'   : loss_segmentation,
                   'disc_A': loss_disc_A,
                   'disc_B': loss_disc_B,
                   'loss_G': loss_G}
        outputs = {'x_AB'  : x_AB,
                   'x_BA'  : x_BA,
                   'x_ABA' : x_ABA,
                   'x_BAB' : x_BAB,
                   'x_AM'  : x_AM}
        return losses, outputs


def _run_mine():
    from torch import nn
    from matplotlib import pyplot as plt
    import time
    
    n_hidden = 400
    n_iter = 4000
    
    # Data parameters.
    N = 10000
    size = 20
    covariance = 0.9
    
    # Covariance matrix.
    cov = np.eye(size*2)
    cov[size:, :size] += np.eye(size)*covariance
    cov[:size, size:] += np.eye(size)*covariance
    
    # Data.
    def sample_data():
        sample = np.random.multivariate_normal(mean=[0]*size*2,
                                               cov=cov,
                                               size=(N,))
        x = sample[:,:size]
        z = sample[:,size:]
        return x, z
    
    # Theoretical mutual information.
    sx = np.linalg.det(cov[:size, :size])
    sz = np.linalg.det(cov[size:, size:])
    s  = np.linalg.det(cov)
    mi_real = 0.5*np.log(sx*sz/s)
    
    # Mutual information estimator.
    class mi_estimation_network(nn.Module):
        def __init__(self, n_hidden):
            super(mi_estimation_network, self).__init__()
            self.n_hidden = n_hidden
            modules = []
            modules.append(nn.Linear(size*2, self.n_hidden))
            #modules.append(nn.SpectralNorm())
            modules.append(nn.ReLU())
            for i in range(0):
                modules.append(nn.Linear(self.n_hidden, self.n_hidden))
                #modules.append(nn.SpectralNorm())
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.n_hidden, 1))
            self.model = nn.Sequential(*tuple(modules))
        
        def forward(self, x, z):
            out = self.model(torch.cat([x, z], dim=-1))
            return out
    
    mi_estimator = mine(mi_estimation_network(n_hidden=n_hidden))
    model = mi_estimator.estimation_network
    
    #def init_orthogonal(m):
        #if isinstance(m, nn.Linear):
            #torch.nn.init.orthogonal_(m.weight)
    #model.apply(init_orthogonal)
    
    # Train
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001,
                                 eps=1e-7,
                                 weight_decay=0.01,
                                 amsgrad=True)
    #optimizer = torch.optim.SGD(params=model.parameters(),
                                #lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
    model.cuda()
    model.train()
    fig, ax = plt.subplots(1,1)
    ax.axhline(mi_real, color='red', linestyle='dashed')
    fig.show()
    fig.canvas.draw()
    loss_history = []
    for i in range(n_iter):
        optimizer.zero_grad()
        x, z = sample_data()
        _, z_marginal = sample_data()
        x = torch.from_numpy(x.astype(np.float32)).cuda()
        z = torch.from_numpy(z.astype(np.float32)).cuda()
        z_marginal = torch.from_numpy(z_marginal.astype(np.float32)).cuda()
        loss = mi_estimator.evaluate(x, z, z_marginal)
                
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        #lr_scheduler.step()
        print("Iteration {} - lower_bound={:.2f} (real {:.2f}) lr={}, norm={}"
              "".format(i, -loss.item(), mi_real, lr_scheduler.get_lr()[0],
                        norm))
        loss_history.append(-loss.item())
        if (i)%100==0:
            plt.scatter(range(i+1), loss_history, c='black', s=2)
            fig.canvas.draw()
            
            def get_sv(layer):
                _, s, _ = np.linalg.svd(layer.weight.data.cpu().numpy())
                return np.mean(s), np.min(s), np.max(s)
            for i, m in enumerate(model.model):
                if isinstance(m, nn.Linear):
                    print("Singular values at layer {}: mean={:.2f}, "
                          "min={:.2f}, max={:.2f}".format(i, *get_sv(m)))
                
            
    plt.show(block=True)    # Keep figure until it's closed.
        
        
def _run_segmentation_model():
    #from fcn_maker import assemble_model
    pass


if __name__=='__main__':
    print("\nRUNNING MINE\n")
    _run_mine()
    print("\nRUNNING SEGMENTATION MODEL\n")
    _run_segmentation_model()
