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
from skimage.transform import resize
import torch
from torch.autograd import Variable
import ignite
from ignite.engines import (Events,
                            Trainer,
                            Evaluator)
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
    g_load.add_argument('--model_from', type=str, default='configs_cyclegan/baseline.py')
    g_load.add_argument('--resume', type=str, default=None)
    parser.add_argument('--vis_freq', type=float, default=0.5)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_pool', type=int, default=0)
    parser.add_argument('--lamb', type=float, default=10.)
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_valid', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def _get_optimizer(name, params, lr):
    if name == 'adam':
        optimizer = torch.optim.Adam(params=params,
                                        lr=lr,
                                        betas=(0.5,0.999),
                                        weight_decay=args.weight_decay)
        return optimizer
    else:
        raise NotImplemented("Optimizer {} not supported."
                            "".format(args.optimizer))

def setup_optimizers(name, g_params, d_a_params, d_b_params, lr):
    optimizer = {
        'g': _get_optimizer(
            args.optimizer,
            g_params,
            args.learning_rate),
        'd_a': _get_optimizer(
            args.optimizer,
            d_a_params,
            args.learning_rate),
        'd_b': _get_optimizer(
            args.optimizer,
            d_b_params,
            args.learning_rate)
    }
    return optimizer

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

    def _process_slice(self, s):
        s = np.squeeze(s)
        s = np.clip(s, 0, 1)
        s[0,0]=1
        s[0,1]=0
        return s

    def __call__(self, engine):

        # Extract images and change them from [-1, 1] to [0,1].
        outputs = engine.state.output[1]
        a, atob, atob_btoa, b, btoa, btoa_atob = \
            [(elem.cpu().numpy()*0.5)+0.5 for elem in outputs[:-1]]
        # Extract the segmentation mask, which is already in Numpy.
        mask = engine.state.output[1][-1]
        
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

        if self._current_batch_num % self.save_every != 0:
            return
            
        # Visualize.
        all_imgs = []
        for i in range(this_batch_size):
            # For each example in the batch, plot all of the
            # channels. So the image will have bs*f rows, where
            # f = 4.
            for j in range(a.shape[1]):
                this_row = []
                this_row.append(a[i,j])
                this_row.append(atob[i,j])
                this_row.append(atob_btoa[i,j])
                this_row.append(resize(b[i,j], a.shape[2:]))
                this_row.append(resize(btoa[i,j], a.shape[2:]))
                this_row.append(resize(btoa_atob[i,j], a.shape[2:]))
                this_row.append(
                    resize(self._process_slice(mask[i]/4.), a.shape[2:]))
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
    data_train = [data['train']['h'], data['train']['s'], data['train']['m']]
    data_valid = [data['valid']['h'], data['valid']['s'], data['train']['m']]
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
                                     batch_size=args.batch_size_train,
                                     preprocessor=preprocessor_train,
                                     nb_io_workers=1,
                                     nb_proc_workers=3,
                                     rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=args.batch_size_valid,
                                     preprocessor=preprocessor_valid,
                                     nb_io_workers=1,
                                     nb_proc_workers=0,
                                     rng=np.random.RandomState(42))

    '''
    Prepare model. The `resume` arg is able to restore the model,
    its state, and the optimizer's state, whereas `model_from`
    (which is mutually exclusive with `resume`) only loads the
    desired architecture.
    '''
    torch.backends.cudnn.benchmark = False   # profiler
    exp_id = None
    if args.resume is not None:
        saved_dict = torch.load(args.resume)
        exp_id = saved_dict['exp_id']
        # Extract the module string, then turn it into a module.
        # From this, we can invoke the model creation function and
        # load its saved weights.
        module_as_str = saved_dict['module_as_str']
        module = imp.new_module('model_from')
        exec(module_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        if not args.cpu:
            for model_ in model.values():
                model_.cuda(args.gpu_id)
        # Load weights for all four networks.
        for key in saved_dict['weights']:
            model[key].load_state_dict(saved_dict['weights'][key])
        # Load optim state for all networks.
        optimizer = setup_optimizers(args.optimizer,
                                     itertools.chain(
                                         model['g_atob'].parameters(),
                                         model['g_btoa'].parameters()),
                                     model['d_a'].parameters(),
                                     model['d_b'].parameters(),
                                     args.learning_rate)
        optimizer['g'].load_state_dict(saved_dict['optim']['g'])
        optimizer['d_a'].load_state_dict(saved_dict['optim']['d_a'])
        optimizer['d_b'].load_state_dict(saved_dict['optim']['d_b'])
    else:
        # If `model_from` is a .py file, then we import that as a module
        # and load the model. Otherwise, we assume it's a pickle, and
        # we load it, extract the module contained inside, and load it
        if args.model_from.endswith(".py"):
            module = imp.load_source('model_from', args.model_from)
            module_as_str = open(args.model_from).read()
            model = getattr(module, 'build_model')()
        else:
            model_dict = torch.load(args.model_from)
            module_as_str = model_dict['module_as_str']
            module = imp.new_module('model_from')
            exec(module_as_str, module.__dict__)
            model = getattr(module, 'build_model')()
            for key in ['g_atob', 'g_btoa', 'd_a', 'd_b']:
                model[key].load_state_dict(model_dict['weights'][key])

        if not args.cpu:
            for model_ in model.values():
                model_.cuda(args.gpu_id)
        optimizer = setup_optimizers(args.optimizer,
                                     itertools.chain(
                                         model['g_atob'].parameters(),
                                         model['g_btoa'].parameters()),
                                     model['d_a'].parameters(),
                                     model['d_b'].parameters(),
                                     args.learning_rate)
    # TODO: do we also save what's inside the
    # image pool?
    fake_A_pool = ImagePool(args.num_pool)
    fake_B_pool = ImagePool(args.num_pool)
    for key in model:
        print("Number of parameters for {}: {}".format(
            key, count_params(model[key])))

    '''
    Set up experiment directory.
    '''
    if exp_id is None:
        exp_id = "{}_{}".format(
            args.name, "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now()))
    path = os.path.join(args.save_path, exp_id)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "cmd.sh"), 'w') as f:
        f.write(' '.join(sys.argv))

    '''
    Set up loss functions.
    '''
    def mse(prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
            target = Variable(target)
        return torch.nn.MSELoss()(prediction, target)

    def compute_g_losses_aba(A_real, atob, atob_btoa):
        """Return all the losses related to generation"""
        atob_gen_loss = mse(model['d_b'](atob), 1)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        return atob_gen_loss, cycle_aba

    def compute_g_losses_bab(B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        btoa_gen_loss = mse(model['d_a'](btoa), 1)
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        return btoa_gen_loss, cycle_bab

    def compute_d_losses(A_real, atob, B_real, btoa):
        """Return all losses related to discriminator"""
        d_a_loss = 0.5*(mse(model['d_a'](A_real), 1) +
                        mse(model['d_a'](fake_A_pool.query(btoa)), 0))
        d_b_loss = 0.5*(mse(model['d_b'](B_real), 1) +
                        mse(model['d_b'](fake_B_pool.query(atob)), 0))
        return d_a_loss, d_b_loss
    
    '''
    Visualize train outputs.
    '''
    epoch_length = lambda ds, bs : len(ds)//bs + int(len(ds)%bs>0)
    num_batches_train = min(
        epoch_length(data['train']['s'], args.batch_size_train),
        epoch_length(data['train']['h'], args.batch_size_train)
    )
    train_save_freq = int(args.vis_freq*num_batches_train)
    image_saver_train = image_saver(save_path=os.path.join(path, "train"),
                                    epoch_length=num_batches_train,
                                    save_every=train_save_freq)

    '''
    Visualize valid outputs.
    '''
    num_batches_valid = min(
        epoch_length(data['valid']['s'], args.batch_size_valid),
        epoch_length(data['valid']['h'], args.batch_size_valid)
    )
    valid_save_freq = int(args.vis_freq*num_batches_valid)
    image_saver_valid = image_saver(save_path=os.path.join(path, "valid"),
                                    epoch_length=num_batches_valid,
                                    save_every=valid_save_freq)

    '''
    Set up training and evaluation functions.
    '''
    def prepare_batch(batch):
        b0, b1, b2 = batch
        # TODO: we norm in [-1, 1] here as this is
        # what cyclegan does
        b0 = Variable(torch.from_numpy(np.array(b0 / 2.)))
        b1 = Variable(torch.from_numpy(np.array(b1 / 2.)))
        if not args.cpu:
            b0 = b0.cuda(args.gpu_id)
            b1 = b1.cuda(args.gpu_id)
        return b0, b1, b2

    def training_function(engine, batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.train()
        # Clear all grad buffers.
        for key in optimizer:
            optimizer[key].zero_grad()
        A_real, B_real, M_np = batch
        both_g_loss = 0.
        # CycleGAN: optimize mapping from A -> B,
        # and from A -> B -> A (cycle).
        hh, ww = A_real.shape[-2], A_real.shape[-1]
        atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
        atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
        # Compute loss for A -> B and cycle.
        atob_gen_loss, cycle_aba = compute_g_losses_aba(
            A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
        both_g_loss += g_tot_loss.item()
        g_tot_loss.backward()
        # CycleGAN: optimize mapping from B -> A,
        # and from B -> A -> B (cycle).
        hh, ww = B_real.shape[-2], B_real.shape[-1]
        btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
        btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
        # Compute loss for B -> A and cycle.
        btoa_gen_loss, cycle_bab = compute_g_losses_bab(
            B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + args.lamb*cycle_bab
        both_g_loss += g_tot_loss.item()
        g_tot_loss.backward()
        # Update discriminator.
        d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        d_a_loss.backward()
        d_b_loss.backward()
        # Update all networks at once.
        for key in optimizer:
            optimizer[key].step()
        this_metrics = {
            'atob_gen_loss': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'btoa_gen_loss': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'd_a_loss': d_a_loss.item(),
            'd_b_loss': d_b_loss.item()
        }
        return both_g_loss, \
            (A_real.data,
             atob.data,
             atob_btoa.data,
             B_real.data,
             btoa.data,
             btoa_atob.data,
             M_np), \
            this_metrics
    trainer = Trainer(training_function)

    def validation_function(engine, batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            A_real, B_real, M_np = batch
            hh, ww = A_real.shape[-2], A_real.shape[-1]
            atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
            atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
            atob_gen_loss, cycle_aba = compute_g_losses_aba(
                A_real, atob, atob_btoa)
            g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
            hh, ww = B_real.shape[-2], B_real.shape[-1]
            btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
            btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
            btoa_gen_loss, cycle_bab = compute_g_losses_bab(
                B_real, btoa, btoa_atob)
            g_tot_loss += btoa_gen_loss + args.lamb*cycle_bab
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        this_metrics = {
            'atob_gen_loss': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'btoa_gen_loss': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'd_a_loss': d_a_loss.item(),
            'd_b_loss': d_b_loss.item()
        }
        return g_tot_loss.item(), \
            (A_real.data,
             atob.data,
             atob_btoa.data,
             B_real.data,
             btoa.data,
             btoa_atob.data,
             M_np), \
            this_metrics
    evaluator = Evaluator(validation_function)

    '''
    Set up logging to screen.
    '''
    progress_train = progress_report(prefix=None,
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, image_saver_train)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(loader_valid))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_valid)

    cpt_handler = ModelCheckpoint(dirname=path,
                                  filename_prefix='weights',
                                  n_saved=5,
                                  save_interval=1,
                                  atomic=False,
                                  exist_ok=True,
                                  create_dir=True,
                                  require_empty=False)
    model_dict = {
        'model': {
            'exp_id': exp_id,
            'weights': {
                'g_atob': model['g_atob'].state_dict(),
                'g_btoa': model['g_btoa'].state_dict(),
                'd_a': model['d_a'].state_dict(),
                'd_b': model['d_b'].state_dict()
            },
            'module_as_str': module_as_str,
            'optim': {
                'g': optimizer['g'].state_dict(),
                'd_a': optimizer['d_a'].state_dict(),
                'd_b': optimizer['d_b'].state_dict()
            }
        }
    }
    evaluator.add_event_handler(Events.COMPLETED,
                                cpt_handler,
                                model_dict)

    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)
