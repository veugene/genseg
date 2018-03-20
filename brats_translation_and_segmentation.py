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
    parser.add_argument('--t_model_from', type=str, default='configs_cyclegan/dilated_fcn.py')
    parser.add_argument('--s_model_from', type=str, default='configs/resunet_hybrid.py')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--cg_coef', type=float, default=1.)
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


'''
Save images on validation.
'''
class image_saver(object):
    def __init__(self, save_path, epoch_length, save_every=1, score_function=None):
        self.save_path = save_path
        self.epoch_length = epoch_length
        self.score_function = score_function
        self.save_every = save_every
        self._max_score = -np.inf
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, engine, state):
        # If tracking a score, only save whenever a max score is reached.
        if self.score_function is not None:
            score = float(self.score_function(state))
            if score > self._max_score:
                self._max_score = score
            else:
                return

        # Unpack inputs, outputs.
        inputs, target, prediction, translation = state.output[1]

        # Current batch size.
        this_batch_size = len(target)

        # Current batch_num, epoch.
        self._current_batch_num += 1

        if self._current_batch_num==self.epoch_length:
            self._current_epoch += 1
            self._current_batch_num = 0

        # Make directory.
        save_dir = os.path.join(self.save_path, str(self._current_epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Only do the image saving if we're at the right
        # interval.
        if self._current_batch_num % self.save_every != 0:
            return
            
        # Variables to numpy.
        inputs = inputs.cpu().numpy()
        target = target.cpu().numpy()
        prediction = prediction.detach().cpu().numpy()
        translation = translation.detach().cpu().numpy()

        # Visualize.
        all_imgs = []
        for i in range(this_batch_size):

            # inputs
            im_i = []
            for x in inputs[i]:
                im_i.append(self._process_slice((x+0.5)*0.5))

            # target
            im_t = [self._process_slice(target[i]/4.)]

            # prediction
            p = prediction[i]
            p[0] = 0
            p[1] *= 1
            p[2] *= 2
            p[3] *= 4
            p = p.max(axis=0)
            im_p = [self._process_slice(p/4.)]

            # translation
            im_tr = []
            for x in translation[i]:
                im_tr.append( x*0.5 + 0.5 )

            out_image = np.concatenate(im_i+im_t+im_p+im_tr, axis=1)
            all_imgs.append(out_image)
        imsave(os.path.join(save_dir,
                            "{}.jpg".format(self._current_batch_num)),
                            np.vstack(all_imgs))

    def _process_slice(self, s):
        s = np.squeeze(s)
        s = np.clip(s, 0, 1)
        s[0,0]=1
        s[0,1]=0
        return s


if __name__ == '__main__':
    args = parse_args()

    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"))
    #data_train_t = [data['train']['h']]
    #data_valid_t = [data['valid']['h']]
    data_train = [data['train']['h'], data['train']['s'], data['train']['m']]
    data_valid = [data['valid']['h'], data['valid']['s'], data['valid']['m']]
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
                                       nb_proc_workers=0,
                                       rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                       sample_random=True,
                                       batch_size=args.batch_size,
                                       preprocessor=preprocessor_valid,
                                       nb_io_workers=1,
                                       nb_proc_workers=0,
                                       rng=np.random.RandomState(42))

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
    image_saver_train_seg = image_saver(save_path=os.path.join(path, "train"),
                                        epoch_length=num_batches_train,
                                        save_every=200)
    
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
    def prepare_batch(batch):
        h, s, m = batch
        # TODO: we norm in [-1, 1] here as this is
        # what cyclegan does
        h = Variable(torch.from_numpy(np.array(h / 2.)))
        s = Variable(torch.from_numpy(np.array(s / 2.)))
        m = Variable(torch.from_numpy(np.array(m)))
        if not args.cpu:
            h = h.cuda(args.gpu_id)
            s = s.cuda(args.gpu_id)
            m = m.cuda(args.gpu_id)
        return h, s, m

    def training_function(batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.train()
        A_real, B_real, M_real = batch
        # CycleGAN: optimise F and back.
        # This is the healthy to sick and back.
        hh, ww = A_real.shape[-2], A_real.shape[-1]
        atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
        atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
        atob_gen_loss, cycle_aba = compute_g_losses_aba(A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
        optimizer['g'].zero_grad()
        g_tot_loss.backward()
        optimizer['g'].step()
        # CycleGAN: optimise G and back.
        # This is the sick to healthy and back.
        hh, ww = B_real.shape[-2], B_real.shape[-1]
        btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
        btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
        btoa_gen_loss, cycle_bab = compute_g_losses_bab(B_real, btoa, btoa_atob)
        # Segmentation: take the real sick and the translated
        # to healthy version, concatenate them, and feed into
        # seg model.
        seg_inp = torch.cat((B_real, btoa), dim=1)
        seg_out = model['seg'](seg_inp)
        seg_loss = 0.
        for i in range(len(loss_functions)):
            seg_loss += loss_functions[i](seg_out, M_real)
        seg_loss /= len(loss_functions) # average
        # TODO: do we need to weight the relative contribution of
        # the segmentation here?
        g_tot_loss = args.cg_coef*(btoa_gen_loss + args.lamb*cycle_bab) + seg_loss
        optimizer['g'].zero_grad()
        optimizer['seg'].zero_grad()
        g_tot_loss.backward()
        optimizer['g'].step()
        optimizer['seg'].step()
        # Update discriminator.
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
        return seg_loss.data.item(), \
            (B_real, M_real, seg_out.detach(), btoa.detach()), \
            this_metrics

    trainer = Trainer(training_function)

    def validation_function(batch):
        batch = prepare_batch(batch)
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            A_real, B_real, M_real = batch
            hh, ww = A_real.shape[-2], A_real.shape[-1]
            atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
            atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
            atob_gen_loss, cycle_aba = compute_g_losses_aba(A_real, atob, atob_btoa)
            g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
            hh, ww = B_real.shape[-2], B_real.shape[-1]
            btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
            btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
            btoa_gen_loss, cycle_bab = compute_g_losses_bab(B_real, btoa, btoa_atob)
            seg_inp = torch.cat((B_real, btoa), dim=1)
            seg_out = model['seg'](seg_inp)
            seg_loss = 0.
            for i in range(len(loss_functions)):
                seg_loss += loss_functions[i](seg_out, M_real)
            seg_loss /= len(loss_functions) # average
            # TODO: do we need to weight the relative contribution of
            # the segmentation here?
            g_tot_loss = args.cg_coef*(btoa_gen_loss + args.lamb*cycle_bab) + seg_loss            
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        this_metrics = OrderedDict({
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        })
        return seg_loss.data.item(), \
            (B_real, M_real, seg_out.detach(), btoa.detach()), \
            this_metrics
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
                'd_b': model['d_b'].state_dict(),
                'seg': model['seg'].state_dict()
            },
            'module_as_str': "module_as_str",
            'optim': {
                'g': optimizer['g'].state_dict(),
                'd_a': optimizer['d_a'].state_dict(),
                'd_b': optimizer['d_b'].state_dict(),
                'seg': optimizer['seg'].state_dict()
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
