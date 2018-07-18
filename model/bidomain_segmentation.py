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
            target = target.cuda(args.gpu_id)
        target = Variable(target)
    return torch.nn.MSELoss()(prediction, target)
    
    
class segmentation_model(object):
    def __init__(self, f_factor, f_common, f_residual, f_unique,
                 g_common, g_residual, g_unique, g_output,
                 disc_A, disc_B, mutual_information, loss_segmentation,
                 z_size=50, z_constant=0, lambda_disc=1, lambda_x_id=10,
                 lambda_z_id=1, lambda_const=1, lambda_cyc=0, lambda_mi=1,
                 lambda_seg=1):
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
        
    def _z_constant(batch_size):
        return Variable(torch.zeros((batch_size, 1, self.z_size),
                                    dtype='float32'))
    
    def _z_sample(batch_size):
        return Variable(torch.randn(batch_size, 1,self.z_size).type('float32'))
        
    def encode(self, x):
        z_a, z_b = self.f_factor(x)
        z_common = self.f_common(z_b)
        z_residual = self.f_residual(z_b)
        z_unique = self.f_unique(z_a)
        z = {'common'  : z_common,
             'residual': z_residual,
             'unique'  : z_unique}
        return z, z_a, z_b
        
    def decode(self, z_common, z_residual, z_unique):
        out = self.g_output(self.g_common(z_common),
                            self.g_residual(z_residual),
                            self.g_unique(z_unique))
        return out
    
    def translate_AB(self, x_A):
        batch_size = len(x_A)
        s_A = self.encode(x_A)
        z_A = {'common'  : s_A['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_constant(batch_size)}
        x_AB = self.decode(**z_A)
        return x_AB
    
    def translate_BA(self, x_B):
        batch_size = len(x_B)
        s_B = self.encode(x_B)
        z_B = {'common'  : s_B['common'],
               'residual': self._z_sample(batch_size),
               'unique'  : self._z_sample(batch_size)}
        x_BA = self.decode(**z_B)
        return x_BA
    
    def segment(self, x_A):
        batch_size = len(x_B)
        s_A = self.encode(x_A)
        z_AM = {'common'  : self._z_constant(batch_size),
                'residual': self._z_constant(batch_size),
                'unique'  : s_A['unique']}
        x_AM = self.decode(**z_AM)
        return x_AM
            
    def update(self, x_A, x_B, mask=None):
        assert len(x_A)==len(x_B)
        batch_size = len(x_A)
        
        # Encode inputs.
        s_A, a_A, b_A = self.encode(x_A)
        s_B, a_B, b_B = self.encode(x_B)
        
        # Reconstruct inputs.
        z_AA = {'common'  : s_A['common'],
                'residual': s_A['residual'],
                'unique'  : s_A['unique']}
        z_BB = {'common'  : s_B['common'],
                'residual': s_B['residual'],
                'unique'  : self._z_constant(batch_size)}
        x_AA = self.decode(**z_AA)
        x_BB = self.decode(**z_BB)
        
        # Translate.
        z_AB = {'common'  : s_A['common'],
                'residual': self._z_sample(batch_size),
                'unique'  : self._z_constant(batch_size)}
        z_BA = {'common'  : s_B['common'],
                'residual': self._z_sample(batch_size),
                'unique'  : self._z_sample(batch_size)}
        x_AB = self.decode(**z_AB)
        x_BA = self.decode(**z_BA)
        
        # Reconstruct latent codes.
        s_AB, a_AB, b_AB = self.encode(x_AB)
        s_BA, a_BA, b_BA = self.encode(x_BA)
        
        # Cycle.
        z_ABA = {'common'  : s_AB['common'],
                 'residual': s_A['residual'],
                 'unique'  : s_A['unique']}
        z_BAB = {'common'  : s_BA['common'],
                 'residual': s_B['residual'],
                 'unique'  : s_B['unique']}
        x_ABA = self.decode(**z_ABA)
        x_BAB = self.decode(**z_BAB)
        
        # Generator losses.
        dist = torch.nn.L1Loss
        loss_discr_AB  = mse(self.disc_B(x_AB), 1)
        loss_discr_BA  = mse(self.disc_A(x_BA), 1)
        loss_recon_AA  = dist(x_AA, x_A)
        loss_recon_BB  = dist(x_BB, x_B)
        loss_recon_zAB = {'common':   dist(s_AB['common'],
                                           z_AB['common'].detach()),
                          'residual': dist(s_AB['residual'],
                                           z_AB['residual']), # detached
                          'unique':   dist(s_AB['unique'],
                                           z_AB['unique'])}   # detached
        loss_recon_zBA = {'common':   dist(s_BA['common'],
                                           z_BA['common'].detach()),
                          'residual': dist(s_BA['residual'],
                                           z_BA['residual']), # detached
                          'unique':   dist(s_BA['unique'],
                                           z_BA['unique'])}   # detached
        loss_const_zB  = dist(s_B['unique'], self._z_constant(batch_size))
        loss_cycle_ABA = dist(x_ABA, x_A)
        loss_cycle_BAB = dist(x_BAB, x_B)
        loss_MI_A      = self.mutual_information.evaluate(a_A, b_A)
        loss_MI_B      = self.mutual_information.evaluate(a_B, b_B)
        loss_MI_AB     = self.mutual_information.evaluate(a_AB, b_AB)
        loss_MI_BA     = self.mutual_information.evaluate(a_BA, b_BA)
        
        # Total generator loss (before segmentation).
        loss_G = (  self.lambda_disc  * (loss_discr_AB+loss_discr_BA)
                  + self.lambda_x_id  * (loss_recon_AA+loss_recon_BB)
                  + self.lambda_z_id  * ( sum(loss_recon_zAB.values())
                                         +sum(loss_recon_zBA.values()))
                  + self.lambda_const * loss_const_zB
                  + self.lambda_cyc   * (loss_cycle_ABA+loss_cycle_BAB)
                  + self.lambda_mi    * ( loss_MI_A+loss_MI_B
                                         +loss_MI_AB+loss_MI_BA))
        
        # Segment.
        if mask is not None:
            z_AM = {'common'  : self._z_constant(batch_size),
                    'residual': self._z_constant(batch_size),
                    'unique'  : s_A['unique']}
            x_AM = self.decode(**z_AM)
            s_AM, a_AM, b_AM = self.encode(x_AM)
            loss_recon_zAM = dist(s_AM['unique'], z_AM['unique'].detach())
            loss_segmentation = self.loss_segmentation(x_AM, mask)
            loss_G += (  self.lambda_seg  * loss_segmentation
                       + self.lambda_z_id * loss_recon_zAM)
        
        # Compute generator gradients.
        loss_G.backward()
        self.disc_A.zero_grad()
        self.disc_B.zero_grad()
        
        # Discriminator losses.
        loss_disc_A = (  mse(self.disc_A(x_A), 1)
                       + mse(self.disc_A(x_BA.detach()), 0))
        loss_disc_B = (  mse(self.disc_B(x_B), 1)
                       + mse(self.disc_B(x_AB.detach()), 0))
        loss_disc_A.backward()
        loss_disc_B.backward()


def _run_mine():
    from torch import nn
    from matplotlib import pyplot as plt
    import time
    
    n_hidden = 400
    n_iter = 10000
    
    # Data parameters.
    N = 10000
    size = 5
    covariance = 0.4
    
    # Covariance matrix.
    cov = np.eye(size*2)
    cov[size:, :size] += np.eye(size)*covariance
    cov[:size, size:] += np.eye(size)*covariance
    
    # Data.
    def sample_data():
        sample = np.random.multivariate_normal(mean=[1]*size*2,
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
            modules.append(nn.ReLU())
            for i in range(1):
                modules.append(nn.Linear(self.n_hidden, self.n_hidden))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.n_hidden, 1))
            self.model = nn.Sequential(*tuple(modules))
        
        def forward(self, x, z):
            out = self.model(torch.cat([x, z], dim=-1))
            return out
    
    mi_estimator = mine(mi_estimation_network(n_hidden=n_hidden))
    model = mi_estimator.estimation_network
    
    # Train
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001,
                                 eps=1e-7,
                                 amsgrad=True)
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
        optimizer.step()
        lr_scheduler.step()
        print("Iteration {} - lower_bound={:.2f} (real {:.2f}) lr={}"
              "".format(i, -loss.item(), mi_real, lr_scheduler.get_lr()[0]))
        loss_history.append(-loss.item())
        if (i+1)%100==0:
            plt.scatter(range(i+1), loss_history, c='black', s=2)
            fig.canvas.draw()
        
        
def _run_segmentation_model():
    #from fcn_maker import assemble_model
    pass


if __name__=='__main__':
    print("\nRUNNING MINE\n")
    _run_mine()
    print("\nRUNNING SEGMENTATION MODEL\n")
    _run_segmentation_model()
