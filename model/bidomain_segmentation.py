import numpy as np
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
        
    def evaluate(x, z):
        permutation = self.rng.permutation(len(z))
        z_shuffled = z[permutation]
        joint = self.estimation_network(x, z)
        marginal = self.estimation_network(x, z_shuffled)
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
                 disc_a, disc_B, mutual_information, loss_segmentation,
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
                 'residual': s_A['residual',
                 'unique'  : s_A['unique']}
        z_BAB = {'common'  : s_BA['common'],
                 'residual': s_B['residual',
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
                                           z_AB['residual'],  # detached
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
