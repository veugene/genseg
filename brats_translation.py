from __future__ import (print_function,
                        division)
from collections import OrderedDict
import sys
import os
import shutil
import argparse
from datetime import datetime
import warnings
import imp

import numpy as np
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import ignite
from ignite.trainer import Trainer
from ignite.engine import Events
from ignite.evaluator import Evaluator
from ignite.handlers import ModelCheckpoint

from architectures.image2image import DilatedFCN
from utils.ignite import (progress_report,
                          metrics_handler,
                          scoring_function)
from utils.metrics import (dice_loss,
                           accuracy)
from utils.data import data_flow_sampler
from utils.data import prepare_data_brats
from utils.data import preprocessor_brats
from util import count_params
import configs
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

from architectures import image2image
from util import ImagePool
import itertools


'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation on BRATS 2017.")
    parser.add_argument('--name', type=str, default="brats_trans")
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    g_load = parser.add_mutually_exclusive_group(required=False)
    g_load.add_argument('--model_from', type=str, default='configs/resunet.py')
    g_load.add_argument('--resume', type=str, default=None)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_pool', type=int, default=0)
    parser.add_argument('--lamb', type=float, default=10.)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def get_optimizer(name, params, lr):
    if name == 'adam':
        optimizer = torch.optim.Adam(params=params,
                                        lr=lr,
                                        betas=(0.5,0.999),
                                        weight_decay=args.weight_decay)
        return optimizer
    else:
        raise NotImplemented("Optimizer {} not supported."
                            "".format(args.optimizer))


'''
Save images on validation.
'''
class image_saver(object):
    def __init__(self, save_path, epoch_length, save_every=1):
        """
        save_every : save every this many minibatches.
        """
        self.save_path = save_path
        self.epoch_length = epoch_length
        self.save_every = save_every
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, engine, state):

        if self._current_batch_num % self.save_every != 0:
            return

        # Extract images and change them from [-1, 1] to [0,1].
        a, atob, atob_btoa, b, btoa, btoa_atob = \
            [ (elem.cpu().numpy()*0.5)+0.5 for elem in state.output[1] ]
        
        # Current batch size.
        this_batch_size = len(a)

        # Current batch_num, epoch.
        self._current_batch_num += 1
        if self._current_batch_num==self.epoch_length:
            self._current_epoch += 1
            self._current_batch_num = 0

        # Make directory.
        save_dir = os.path.join(self.save_path, str(self._current_epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from skimage.transform import resize
            
        # Visualize.
        all_imgs = []
        for i in range(this_batch_size):

            this_row = []
            # TODO: just show one channel only first
            # Images in B may not be the same size so do a
            # resize if necessary.
            this_row.append(a[i,0])
            this_row.append(atob[i,0])
            this_row.append(atob_btoa[i,0])
            this_row.append( resize(b[i,0], a.shape[2:]) )
            this_row.append( resize(btoa[i,0], a.shape[2:]) )
            this_row.append( resize(btoa_atob[i,0], a.shape[2:]) )
            #out_image = np.concatenate(im_i+im_t+im_p, axis=1)
            out_image = np.hstack(this_row)
            all_imgs.append(out_image)
        imsave(os.path.join(save_dir,
                            "{}.png".format(self._current_batch_num)),
                            np.vstack(all_imgs))

    
if __name__ == '__main__':
    args = parse_args()

    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"))
    data_train = [data['train']['s'], data['train']['h']]
    data_valid = [data['valid']['s'], data['valid']['h']]
    # Prepare data augmentation and data loaders.
    da_kwargs = {'rotation_range': 3.,
                 'zoom_range': 0.1,
                 'horizontal_flip': True,
                 'vertical_flip': True,
                 'spline_warp': True,
                 'warp_sigma': 5,
                 'warp_grid_size': 3}
    preprocessor_train = preprocessor_brats(data_augmentation_kwargs=da_kwargs)
    loader_train = data_flow_sampler(data_train,
                                     sample_random=True,
                                     batch_size=args.batch_size,
                                     preprocessor=preprocessor_train,
                                     nb_io_workers=1,
                                     nb_proc_workers=3,
                                     rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=args.batch_size,
                                     preprocessor=preprocessor_valid,
                                     nb_io_workers=1,
                                     nb_proc_workers=0,
                                     rng=np.random.RandomState(42))

    '''
    hs = []
    ws = []
    c = 0
    for batch in loader_train.flow():
        hs.append(batch[0].shape[2])
        ws.append(batch[0].shape[3])
        hs.append(batch[1].shape[2])
        ws.append(batch[1].shape[3])
        c += 1
        print(c)
        if c == 1000:
            break
    import pdb
    pdb.set_trace()
    '''

    '''
    Prepare model. The `resume` arg is able to restore the model,
    its state, and the optimizer's state, whereas `model_from`
    (which is mutually exclusive with `resume`) only loads the
    desired architecture.
    '''
    torch.backends.cudnn.benchmark = False   # profiler
    exp_id = None
    disc_params = {'input_nc':4, 'ndf':64, 'n_layers_D':3, 'norm':'instance', 'which_model_netD':'n_layers'}
    model = {
        'g_atob': image2image.define_G(4,4,64,'resnet_9blocks', norm='instance'),
        'g_btoa': image2image.define_G(4,4,64,'resnet_9blocks', norm='instance'),
        'd_a': image2image.define_D(**disc_params),
        'd_b': image2image.define_D(**disc_params)
    }
    if not args.cpu:
        for model_ in model.values():
            model_.cuda()
    fake_A_pool = ImagePool(args.num_pool)
    fake_B_pool = ImagePool(args.num_pool)
    optimizer = {
        'g': get_optimizer(
            args.optimizer,
            itertools.chain(model['g_atob'].parameters(), model['g_btoa'].parameters()),
            args.learning_rate
        ),
        'd_a': get_optimizer(args.optimizer, model['d_a'].parameters(), args.learning_rate),
        'd_b': get_optimizer(args.optimizer, model['d_b'].parameters(), args.learning_rate)
    }


    '''
    Set up experiment directory.
    '''
    if exp_id is None:
        exp_id = "{}_{}".format(args.name, "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now()))
    path = os.path.join(args.save_path, exp_id)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "cmd.sh"), 'w') as f:
        f.write(' '.join(sys.argv))

    def compute_g_losses_aba(A_real, atob, atob_btoa):
        """Return all the losses related to generation"""
        g_atob, g_btoa, d_a, d_b = model['g_atob'], model['g_btoa'], model['d_a'], model['d_b']
        d_b_fake = d_b(atob)
        ones_db = torch.ones(d_b_fake.size())
        if not args.cpu:
            ones_db = ones_db.cuda()
        ones_db = Variable(ones_db)
        atob_gen_loss = mse_loss(d_b_fake, ones_db)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        return atob_gen_loss, cycle_aba

    def compute_g_losses_bab(B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        g_atob, g_btoa, d_a, d_b = model['g_atob'], model['g_btoa'], model['d_a'], model['d_b']
        d_a_fake = d_a(btoa)
        ones_da = torch.ones(d_a_fake.size())
        if not args.cpu:
            ones_da = ones_da.cuda()
        ones_da = Variable(ones_da)
        btoa_gen_loss = mse_loss(d_a_fake, ones_da)
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        return btoa_gen_loss, cycle_bab
    
    def compute_d_losses(A_real, atob, B_real, btoa):
        """Return all losses related to discriminator"""
        g_atob, g_btoa, d_a, d_b = model['g_atob'], model['g_btoa'], model['d_a'], model['d_b']
        d_a_real = d_a(A_real)
        d_b_real = d_b(B_real)
        A_fake = fake_A_pool.query(btoa)
        B_fake = fake_B_pool.query(atob)
        d_a_fake = d_a(A_fake)
        d_b_fake = d_b(B_fake)
        ones_da_real, zeros_da_fake = torch.ones(d_a_real.size()), torch.zeros(d_a_fake.size())
        ones_db_real, zeros_db_fake = torch.ones(d_b_real.size()), torch.zeros(d_b_fake.size())
        if not args.cpu:
            ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                        ones_da_real.cuda(), zeros_da_fake.cuda(), ones_db_real.cuda(), zeros_db_fake.cuda()
        ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                        Variable(ones_da_real), Variable(zeros_da_fake), Variable(ones_db_real), Variable(zeros_db_fake)
        d_a_loss = (mse_loss(d_a_real, ones_da_real) + mse_loss(d_a_fake, zeros_da_fake)) * 0.5
        d_b_loss = (mse_loss(d_b_real, ones_db_real) + mse_loss(d_b_fake, zeros_db_fake)) * 0.5
        return d_a_loss, d_b_loss


    '''
    Set up loss function.
    '''
    mse_loss = torch.nn.MSELoss()
    if not args.cpu:
        mse_loss = mse_loss.cuda()
    '''
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
    '''
    '''
    labels = [0,1,2,4] # 4 classes
    loss_functions = []
    metrics_dict = OrderedDict()
    for idx,l in enumerate(labels):
        dice = dice_loss(l,idx)
        if not args.cpu:
            dice = dice.cuda(args.gpu_id)
        loss_functions.append(dice)
        metrics_dict['dice{}'.format(l)] = dice
    metrics = metrics_handler(metrics_dict)
    '''
    
    '''
    Visualize train outputs.
    '''
    bs = args.batch_size
    epoch_length = lambda ds : len(ds)//bs + int(len(ds)%bs>0)
    num_batches_train = min(
        epoch_length(data['train']['s']),
        epoch_length(data['train']['h'])
    )
    image_saver_train = image_saver(save_path=os.path.join(path, "train"),
                                    epoch_length=num_batches_train)

    '''
    Visualize valid outputs.
    '''
    num_batches_valid = min(
        epoch_length(data['valid']['s']),
        epoch_length(data['valid']['h'])
    )
    image_saver_valid = image_saver(save_path=os.path.join(path, "valid"),
                                    epoch_length=num_batches_valid)
    
    '''
    Set up training and evaluation functions.
    '''
    def prepare_batch(batch):
        b0, b1 = batch
        # TODO: we norm in [-1, 1] here as this is
        # what cyclegan does
        b0 = Variable(torch.from_numpy(np.array(b0 / 2.)))
        b1 = Variable(torch.from_numpy(np.array(b1 / 2.)))
        if not args.cpu:
            b0 = b0.cuda(args.gpu_id)
            b1 = b1.cuda(args.gpu_id)
        return b0, b1

    def training_function(batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.train()
        A_real, B_real = batch
        # optimise F and back
        hh, ww = A_real.shape[-2], A_real.shape[-1]
        atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
        atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
        atob_gen_loss, cycle_aba = compute_g_losses_aba(A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
        optimizer['g'].zero_grad()
        g_tot_loss.backward()
        optimizer['g'].step()
        # optimise G and back
        hh, ww = B_real.shape[-2], B_real.shape[-1]
        btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
        btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
        btoa_gen_loss, cycle_bab = compute_g_losses_bab(B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + args.lamb*cycle_bab
        optimizer['g'].zero_grad()
        g_tot_loss.backward()
        optimizer['g'].step()
        # update discriminator
        d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        optimizer['d_a'].zero_grad()
        d_a_loss.backward()
        optimizer['d_a'].step()
        optimizer['d_b'].zero_grad()
        d_b_loss.backward()
        optimizer['d_b'].step()
        this_metrics = {
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        }
        return g_tot_loss.data.item(), (A_real.data, atob.data, atob_btoa.data, B_real.data, btoa.data, btoa_atob.data), this_metrics
    trainer = Trainer(training_function)

    def validation_function(batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            A_real, B_real = batch
            hh, ww = A_real.shape[-2], A_real.shape[-1]
            atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
            atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
            atob_gen_loss, cycle_aba = compute_g_losses_aba(A_real, atob, atob_btoa)
            g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
            hh, ww = B_real.shape[-2], B_real.shape[-1]
            btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
            btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
            btoa_gen_loss, cycle_bab = compute_g_losses_bab(B_real, btoa, btoa_atob)
            g_tot_loss = btoa_gen_loss + args.lamb*cycle_bab
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        this_metrics = {
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        }
        return g_tot_loss.data.item(), (A_real.data, atob.data, atob_btoa.data, B_real.data, btoa.data, btoa_atob.data), this_metrics
    evaluator = Evaluator(validation_function)

    '''
    Set up logging to screen.
    '''
    bs = args.batch_size
    progress_train = progress_report(prefix=None,
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, image_saver_train)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda engine, state: evaluator.run(loader_valid))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_valid)

    '''
    cpt_handler = ModelCheckpoint(dirname=path,
                                  filename_prefix='weights',
                                  n_saved=5,
                                  score_function=scoring_function("val_metrics"),
                                  atomic=False,
                                  exist_ok=True,
                                  create_dir=True)
    evaluator.add_event_handler(Events.COMPLETED,
                                cpt_handler,
                                {'model':
                                 {'exp_id': exp_id,
                                  'weights': model.state_dict(),
                                  'module_as_str': module_as_str,
                                  'optim': optimizer.state_dict()}
                                })
    '''

    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)
