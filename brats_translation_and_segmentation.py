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
    g_load.add_argument('--t_model_from', type=str, default='configs_cyclegan/baseline.py')
    g_load.add_argument('--s_model_from', type=str, default='configs/resunet_hybrid.py')
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

def get_optimizers(name, g_params, d_a_params, d_b_params, seg_params, lr):
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
            args.learning_rate),
        'seg': _get_optimizer(
            args.optimizer,
            seg_params,
            args.learning_rate)
    }
    return optimizer

from brats_translation import image_saver as image_saver_t
from brats_segmentation import image_saver as image_saver_s

if __name__ == '__main__':
    args = parse_args()

    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"))
    data_train_t = [data['train']['s'], data['train']['h']]
    data_valid_t = [data['valid']['s'], data['valid']['h']]
    data_train_s = [data['train']['s'], data['train']['m']]
    data_valid_s = [data['valid']['s'], data['valid']['m']]
    # Prepare data augmentation and data loaders.
    da_kwargs = {'rotation_range': 3.,
                 'zoom_range': 0.1,
                 'horizontal_flip': True,
                 'vertical_flip': True,
                 'spline_warp': True,
                 'warp_sigma': 5,
                 'warp_grid_size': 3}
    preprocessor_train = preprocessor_brats(data_augmentation_kwargs=da_kwargs)
    loader_train_t = data_flow_sampler(data_train_t,
                                       sample_random=True,
                                       batch_size=args.batch_size,
                                       preprocessor=preprocessor_train,
                                       nb_io_workers=1,
                                       nb_proc_workers=3,
                                       rng=np.random.RandomState(42))
    loader_train_s = data_flow_sampler(data_train_s,
                                       sample_random=True,
                                       batch_size=args.batch_size,
                                       preprocessor=preprocessor_train,
                                       nb_io_workers=1,
                                       nb_proc_workers=3,
                                       rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid_t = data_flow_sampler(data_valid_t,
                                       sample_random=True,
                                       batch_size=args.batch_size,
                                       preprocessor=preprocessor_valid,
                                       nb_io_workers=1,
                                       nb_proc_workers=0,
                                       rng=np.random.RandomState(42))
    loader_valid_s = data_flow_sampler(data_valid_s,
                                       sample_random=True,
                                       batch_size=args.batch_size,
                                       preprocessor=preprocessor_valid,
                                       nb_io_workers=1,
                                       nb_proc_workers=0,
                                       rng=np.random.RandomState(42))

    loader_train = zip(loader_train_t.flow(), loader_train_s.flow())
    loader_valid = zip(loader_valid_t.flow(), loader_valid_s.flow())

    bs = args.batch_size
    epoch_length = lambda ds : len(ds)//bs + int(len(ds)%bs>0)
    
    '''
    Prepare model. The `resume` arg is able to restore the model,
    its state, and the optimizer's state, whereas `model_from`
    (which is mutually exclusive with `resume`) only loads the
    desired architecture.
    '''
    torch.backends.cudnn.benchmark = False   # profiler
    exp_id = None
    if args.resume is not None:
        '''
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
        optimizer = get_optimizers(args.optimizer,
                                   itertools.chain(
                                       model['g_atob'].parameters(),
                                       model['g_btoa'].parameters()),
                                   model['d_a'].parameters(),
                                   model['d_b'].parameters(),
                                   args.learning_rate)
        optimizer['g'].load_state_dict(saved_dict['optim']['g'])
        optimizer['d_a'].load_state_dict(saved_dict['optim']['d_a'])
        optimizer['d_b'].load_state_dict(saved_dict['optim']['d_b'])
        '''
        raise NotImplementedError("")
    else:
        # If `model_from` is a .py file, then we import that as a module
        # and load the model. Otherwise, we assume it's a pickle, and
        # we load it, extract the module contained inside, and load it

        # Process translation model.
        if args.t_model_from.endswith(".py"):
            t_module = imp.load_source('t_model_from', args.t_model_from)
            t_module_as_str = open(args.t_model_from).read()
        else:
            t_module_as_str = torch.load(args.t_model_from)['module_as_str']
            t_module = imp.new_module('t_model_from')
            exec(t_module_as_str, t_module.__dict__)
        # Process segmentation model.
        if args.s_model_from.endswith(".py"):
            s_module = imp.load_source('s_model_from', args.s_model_from)
            s_module_as_str = open(args.s_model_from).read()
        else:
            s_module_as_str = torch.load(args.s_model_from)['module_as_str']
            s_module = imp.new_module('s_model_from')
            exec(s_module_as_str, s_module.__dict__)
        # Translation model is a dict, so do this first,
        # then plop in the segmentation model.
        model = getattr(t_module, 'build_model')()
        model['seg'] = getattr(s_module, 'build_model')()
        if not args.cpu:
            for model_ in model.values():
                model_.cuda(args.gpu_id)
        optimizer = get_optimizers(args.optimizer,
                                   itertools.chain(
                                       model['g_atob'].parameters(),
                                       model['g_btoa'].parameters()),
                                   model['d_a'].parameters(),
                                   model['d_b'].parameters(),
                                   model['seg'].parameters(),
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
            ones_db = ones_db.cuda(args.gpu_id)
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
            ones_da = ones_da.cuda(args.gpu_id)
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
                        ones_da_real.cuda(args.gpu_id), \
                        zeros_da_fake.cuda(args.gpu_id), \
                        ones_db_real.cuda(args.gpu_id), \
                        zeros_db_fake.cuda(args.gpu_id)
        ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                        Variable(ones_da_real), \
                        Variable(zeros_da_fake), \
                        Variable(ones_db_real), \
                        Variable(zeros_db_fake)
        d_a_loss = (mse_loss(d_a_real, ones_da_real) + mse_loss(d_a_fake, zeros_da_fake)) * 0.5
        d_b_loss = (mse_loss(d_b_real, ones_db_real) + mse_loss(d_b_fake, zeros_db_fake)) * 0.5
        return d_a_loss, d_b_loss

    '''
    Set up loss function.
    '''
    mse_loss = torch.nn.MSELoss()
    if not args.cpu:
        mse_loss = mse_loss.cuda(args.gpu_id)

    '''
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
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
    Visualize train outputs.
    '''
    num_batches_train = min(
        epoch_length(data['train']['s']),
        epoch_length(data['train']['h'])
    )
    image_saver_train_seg = image_saver_s(save_path=os.path.join(path, "train"),
                                    epoch_length=num_batches_train,
                                    save_every=500)
    #image_saver_train_trans = image_saver_t(save_path=os.path.join(path, "train"),
    #                                epoch_length=num_batches_train,
    #                                save_every=500)
    
    '''
    Visualize valid outputs.
    '''
    '''
    num_batches_valid = min(
        epoch_length(data['valid']['s']),
        epoch_length(data['valid']['h'])
    )
    image_saver_valid = image_saver_(save_path=os.path.join(path, "valid"),
                                    epoch_length=num_batches_valid,
                                    save_every=500)
    '''
    
    '''
    Set up training and evaluation functions for translation.
    '''
    def prepare_batch_translation(batch):
        b0, b1 = batch
        # TODO: we norm in [-1, 1] here as this is
        # what cyclegan does
        b0 = Variable(torch.from_numpy(np.array(b0 / 2.)))
        b1 = Variable(torch.from_numpy(np.array(b1 / 2.)))
        if not args.cpu:
            b0 = b0.cuda(args.gpu_id)
            b1 = b1.cuda(args.gpu_id)
        return b0, b1

    def training_function_translation(batch):
        batch = prepare_batch_translation(batch)
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
        return g_tot_loss.data.item(), \
            (A_real.data, atob.data, atob_btoa.data, B_real.data, btoa.data, btoa_atob.data), \
            this_metrics

    def validation_function_translation(batch):
        batch = prepare_batch_translation(batch)
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
        this_metrics = OrderedDict({
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        })
        return g_tot_loss.data.item(), \
            (A_real.data, atob.data, atob_btoa.data, B_real.data, btoa.data, btoa_atob.data), \
            this_metrics

    '''
    Set up functions for training segmentation.
    '''
    def prepare_batch_segmentation(batch):
        b0, b1 = batch
        b0 = Variable(torch.from_numpy(np.array(b0 / 2.)))
        b1 = Variable(torch.from_numpy(np.array(b1)))
        if not args.cpu:
            b0 = b0.cuda(args.gpu_id)
            b1 = b1.cuda(args.gpu_id)
        return b0, b1

    def training_function_segmentation(batch):
        """
        s --> h -----> m
              s --|
        """
        batch = prepare_batch_segmentation(batch)
        for model_ in model.values():
            model_.train()
        optimizer['seg'].zero_grad()
        optimizer['g'].zero_grad() ##
        hh, ww = batch[0].shape[-2], batch[0].shape[-1]
        s2h = model['g_atob'](batch[0])[:, :, 0:hh, 0:ww]
        inp_concat = torch.cat((batch[0], s2h), dim=1)
        output = model['seg'](inp_concat)
        loss = 0.
        for i in range(len(loss_functions)):
            loss += loss_functions[i](output, batch[1])
        loss /= len(loss_functions) # average
        loss.backward()
        optimizer['seg'].step()
        optimizer['g'].step() ##
        return loss.data.item(), \
            (batch[0], batch[1], output), \
            metrics(output, batch[1])

    def validation_function_segmentation(batch):
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            batch = prepare_batch_segmentation(batch)
            hh, ww = batch[0].shape[-2], batch[0].shape[-1]
            s2h = model['g_atob'](batch[0])[:, :, 0:hh, 0:ww]
            inp_concat = torch.cat((batch[0], s2h), dim=1)
            output = model['seg'](inp_concat)
            loss = 0.
            for i in range(len(loss_functions)):
                loss += loss_functions[i](output, batch[1])
            loss /= len(loss_functions) # average
        return loss.data.item(), \
            (batch[0], batch[1], output), \
            metrics(output, batch[1])

    def training_function(batch):
        batch_t, batch_s = batch
        # batch_t contains (sick,healthy)
        # and batch_s contains (sick,mask)
        batch_t = prepare_batch_translation(batch_t)
        batch_s = prepare_batch_segmentation(batch_s)
        # Handle the translation part.
        t_loss, t_vis, t_metrics = training_function_translation(batch_t)
        # Handle the segmentation part.
        s_loss, s_vis, s_metrics = training_function_segmentation(batch_s)
        # TODO??
        #s_metrics.update(t_metrics)
        s_metrics['g_atob'] = t_metrics['atob_gen_loss']
        s_metrics['d_b'] = t_metrics['d_b_loss']
        return s_loss, s_vis, s_metrics
    trainer = Trainer(training_function)

    def validation_function(batch):
        batch_t, batch_s = batch
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            batch_t = prepare_batch_translation(batch_t)
            batch_s = prepare_batch_segmentation(batch_s)
            # Handle the translation part.
            t_loss, t_vis, t_metrics = validation_function_translation(batch_t)
            # Handle the segmentation part.
            s_loss, s_vis, s_metrics = validation_function_segmentation(batch_s)
            #s_metrics.update(t_metrics)
            s_metrics['g_atob'] = t_metrics['atob_gen_loss']
            s_metrics['d_b'] = t_metrics['d_b_loss']
        # TODO??
        return s_loss, s_vis, s_metrics
    evaluator = Evaluator(validation_function)

    '''
    Set up logging to screen.
    '''
    progress_train = progress_report(prefix=None,
                                     epoch_length=min(
                                         epoch_length(data['train']['h']),
                                         epoch_length(data['train']['s'])
                                     ),
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     epoch_length=min(
                                         epoch_length(data['valid']['h']),
                                         epoch_length(data['valid']['s'])
                                     ),
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, image_saver_train_seg)
    #trainer.add_event_handler(Events.ITERATION_COMPLETED, image_saver_train_trans)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda engine, state: evaluator.run(loader_valid))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    #evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_valid)

    '''
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

    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)
