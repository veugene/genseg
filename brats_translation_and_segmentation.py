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
from utils.data import (data_flow_sampler,
                        preprocessor_brats)
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
    parser.add_argument('--dataset', type=str, default='brats17')
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    # TODO: do proper mutual exclusion
    parser.add_argument('--t_model_from', type=str,
                        default='configs_cyclegan/dilated_fcn.py')
    parser.add_argument('--s_model_from', type=str,
                        default='configs/resunet_hybrid.py')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--vis_freq', type=float, default=0.5)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--masked_fraction', type=float, default=0)
    parser.add_argument('--orientation', type=int, default=None)
    parser.add_argument('--num_pool', type=int, default=0)
    parser.add_argument('--lamb', type=float, default=10.)
    parser.add_argument('--detach', action='store_true', default=False)
    parser.add_argument('--no_valid', default=False, action='store_true')
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_valid', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--nb_io_workers', type=int, default=1)
    parser.add_argument('--nb_proc_workers', type=int, default=0)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--rseed', type=int, default=42)
    args = parser.parse_args()
    return args


def _get_optimizer(name, params, lr):
    if name == 'adam':
        optimizer = torch.optim.Adam(params=params,
                                        lr=lr,
                                        betas=(0.5,0.999),
                                        weight_decay=args.weight_decay)
        return optimizer
    elif name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=params,
                                        lr=lr,
                                        alpha=0.9,
                                        weight_decay=args.weight_decay)
        return optimizer
    else:
        raise NotImplemented("Optimizer {} not supported."
                            "".format(args.optimizer))

def setup_optimizers(name, g_params, d_a_params, d_b_params, seg_params, lr):
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
    def __init__(self, save_path, epoch_length, save_every=1,
                 score_function=None):
        self.save_path = save_path
        self.epoch_length = epoch_length
        self.score_function = score_function
        self.save_every = save_every
        self._max_score = -np.inf
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, engine):

        # Current batch_num, epoch.
        self._current_batch_num += 1

        if self._current_batch_num==self.epoch_length:
            self._current_epoch += 1
            self._current_batch_num = 0
        
        # If tracking a score, only save whenever a max score is reached.
        if self.score_function is not None:
            score = float(self.score_function(engine.state))
            if score > self._max_score:
                self._max_score = score
            else:
                return

        # Unpack inputs, outputs.
        inputs, target_, prediction_, translation, cycle, indices = \
            engine.state.output[2]
        
        # Current batch size.
        this_batch_size = len(inputs)

        if this_batch_size == 0:
            return

        # Make directory.
        save_dir = os.path.join(self.save_path, str(self._current_epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Only do the image saving if we're at the right
        # interval.
        if self._current_batch_num % self.save_every != 0:
            return
            
        # Variables to numpy.
        inputs = inputs.cpu().numpy() # full
        target_ = target_.cpu().numpy() # label only
        target = [np.zeros_like(inputs[0][0])]*this_batch_size
        for i in range(len(indices)):
            target[indices[i]] = target_[i]
        target = np.asarray(target)
        
        prediction_ = prediction_.cpu().numpy() # label only
        prediction = [np.zeros_like(inputs[0])]*this_batch_size
        for i in range(len(indices)):
            prediction[indices[i]] = prediction_[i]
        prediction = np.asarray(prediction)
        
        translation = translation.cpu().numpy() # full
        cycle = cycle.cpu().numpy() # full

        # Visualize.
        all_imgs = []
        for i in range(this_batch_size):

            # inputs
            im_i = []
            for x in inputs[i]:
                im_i.append(self._process_slice(x*0.5 + 0.5))

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
                im_tr.append(x*0.5 + 0.5)

            # cycle
            im_cyc = []
            for x in cycle[i]:
                im_cyc.append(self._process_slice(x*0.5 + 0.5))
                
            out_image = np.concatenate(im_i+im_t+im_p+im_tr+im_cyc,
                                       axis=1)
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

    if args.dataset not in ['brats17', 'brats13s']:
        raise Exception("Dataset must be either brats17 or " +
                        "brats13")
    else:
        if args.dataset == 'brats17':
            from utils.data import prepare_data_brats17 as \
                prepare_data_brats
        else:
            from utils.data import prepare_data_brats13s as \
                prepare_data_brats

    orientation = None
    if type(args.orientation) == int:
        orientation = [args.orientation]
    
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"),
                              orientations=orientation,
                              masked_fraction=args.masked_fraction,
                              rng=np.random.RandomState(args.rseed))
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
                                     batch_size=args.batch_size_train,
                                     preprocessor=preprocessor_train,
                                     nb_io_workers=args.nb_io_workers,
                                     nb_proc_workers=args.nb_proc_workers,
                                     rng=np.random.RandomState(args.rseed))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=args.batch_size_valid,
                                     preprocessor=preprocessor_valid,
                                     nb_io_workers=args.nb_io_workers,
                                     nb_proc_workers=args.nb_proc_workers,
                                     rng=np.random.RandomState(args.rseed))

    epoch_length = lambda ds, bs : len(ds)//bs + int(len(ds)%bs>0)
    
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
        t_module_as_str = saved_dict['t_module_as_str']
        s_module_as_str = saved_dict['s_module_as_str']
        # Load the translation module contained
        # in the dict.
        t_module = imp.new_module('t_model_from')
        exec(t_module_as_str, t_module.__dict__)
        model = getattr(t_module, 'build_model')()
        # Load the segmentation model contained
        # in the dict.
        s_module = imp.new_module('s_model_from')
        exec(s_module_as_str, s_module.__dict__)
        s_model = getattr(s_module, 'build_model')()
        model['seg'] = s_model
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
                                   model['seg'].parameters(),
                                   args.learning_rate)
        optimizer['g'].load_state_dict(saved_dict['optim']['g'])
        optimizer['d_a'].load_state_dict(saved_dict['optim']['d_a'])
        optimizer['d_b'].load_state_dict(saved_dict['optim']['d_b'])
    else:
        # If `model_from` is a .py file, then we import that as a module
        # and load the model. Otherwise, we assume it's a checkpoint, and
        # we load it, extract the modules contained inside, and load the
        # translation + seg models.

        # Process translation model.
        model = {}
        if args.t_model_from.endswith(".py"):
            t_module = imp.load_source('t_model_from', args.t_model_from)
            t_module_as_str = open(args.t_model_from).read()
        else:
            t_model_dict = torch.load(args.t_model_from)
            t_module_as_str = t_model_dict['t_module_as_str']
            t_module = imp.new_module('t_model_from')
            exec(t_module_as_str, t_module.__dict__)
            # Return the models and load in their weights.
            t_models = getattr(t_module, 'build_model')()
            model.update(t_models)
            for key in ['g_atob', 'g_btoa', 'd_a', 'd_b']:
                model[key].load_state_dict(t_model_dict['weights'][key])
        # Process segmentation model.
        if args.s_model_from.endswith(".py"):
            s_module = imp.load_source('s_model_from', args.s_model_from)
            s_module_as_str = open(args.s_model_from).read()
        else:
            s_model_dict = torch.load(args.s_model_from)
            s_module_as_str = s_model_dict['s_module_as_str']
            s_module = imp.new_module('s_model_from')
            exec(s_module_as_str, s_module.__dict__)
            # Return the models and load in their weights.
            model['seg'] = getattr(s_module, 'build_model')()
            model['seg'].load_state_dict(s_model_dict['weights']['seg'])
        # Translation model is a dict, so do this first,
        # then plop in the segmentation model.
        model = getattr(t_module, 'build_model')()
        model['seg'] = getattr(s_module, 'build_model')()

        if not args.cpu:
            for model_ in model.values():
                model_.cuda(args.gpu_id)
        optimizer = setup_optimizers(args.optimizer,
                                     itertools.chain(
                                       model['g_atob'].parameters(),
                                       model['g_btoa'].parameters()),
                                     model['d_a'].parameters(),
                                     model['d_b'].parameters(),
                                     model['seg'].parameters(),
                                     args.learning_rate)

    # Note: currently, contents of the image pool
    # are not saved with the model.
    fake_A_pool = ImagePool(args.num_pool)
    fake_B_pool = ImagePool(args.num_pool)
    for key in model:
        print("Number of parameters for {}: {}".format(
            key, count_params(model[key])))

    '''
    Set up experiment directory.
    '''
    if exp_id is None:
        exp_id = args.name
        if not args.no_timestamp:
            exp_id += "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
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
                target = target.cuda(args.gpu_id)
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
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
    '''
    labels = [0,1,2,4] # 4 classes
    loss_functions = []
    metrics = {'train': None, 'valid': None}
    for key in metrics.keys():
        metrics_dict = OrderedDict()
        
        # Dice score for every class.
        for idx,l in enumerate(labels):
            dice = dice_loss(l,idx)
            g_dice = dice_loss(target_class=l, target_index=idx,
                               accumulate=True)
            if not args.cpu:
                dice = dice.cuda(args.gpu_id)
                g_dice = g_dice.cuda(args.gpu_id)
            loss_functions.append(dice)
            metrics_dict['dice{}'.format(l)] = g_dice
            
        # Overall tumour Dice.
        g_dice = dice_loss(target_class=[1,2,4], target_index=[1,2,3],
                           accumulate=True)
        if not args.cpu:
            g_dice = g_dice.cuda(args.gpu_id)
        metrics_dict['dice124'] = g_dice
        
        metrics[key] = metrics_handler(metrics_dict)
       
    '''
    Visualize train outputs.
    '''
    num_batches_train = min(
        epoch_length(data['train']['s'], args.batch_size_train),
        epoch_length(data['train']['h'], args.batch_size_train)
    )
    train_save_freq = int(np.ceil(args.vis_freq*num_batches_train))
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
    valid_save_freq = int(np.ceil(args.vis_freq*num_batches_valid))
    image_saver_valid = image_saver(save_path=os.path.join(path, "valid"),
                                    epoch_length=num_batches_valid,
                                    save_every=valid_save_freq)

    '''
    Set up training and evaluation functions for translation.
    '''
    def prepare_batch(batch):
        h, s, m = batch
        # Identify which segmentation masks are provided, if any.
        indices = [i for i in range(len(m)) if m[i] is not None]
        m = [m[i] for i in indices]
        # Scaling differs from segmentation-only task in
        # that we divide h,s by 2 here, so it's in [-1,1].
        h = Variable(torch.from_numpy(np.array(h)/2.))
        s = Variable(torch.from_numpy(np.array(s)/2.))
        m = Variable(torch.from_numpy(np.array(m)))
        if not args.cpu:
            h = h.cuda(args.gpu_id)
            s = s.cuda(args.gpu_id)
            m = m.cuda(args.gpu_id)
        return h, s, m, indices

    def training_function(engine, batch):
        A_real, B_real, M_real, indices = prepare_batch(batch)
        for model_ in model.values():
            model_.train()
        
        # Clear all grad buffers.
        for key in optimizer:
            optimizer[key].zero_grad()
        # CycleGAN: optimize mapping from A -> B,
        # and from A -> B -> A (cycle).
        hh, ww = A_real.shape[-2], A_real.shape[-1]
        atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
        atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
        # Compute loss for A -> B and cycle.
        atob_gen_loss, cycle_aba = compute_g_losses_aba(A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + args.lamb*cycle_aba
        g_tot_loss.backward()
        # CycleGAN: optimize mapping from B -> A,
        # and from B -> A -> B (cycle).
        hh, ww = B_real.shape[-2], B_real.shape[-1]
        btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
        btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
        # Compute loss for B -> A and cycle.
        btoa_gen_loss, cycle_bab = compute_g_losses_bab(B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + args.lamb*cycle_bab
        # Segmentation: take the real sick and the translated
        # to healthy version, concatenate them, and feed into
        # seg model. If 'detach' mode is enabled, then gradients
        # from the seg model do not feed back into the generator.
        seg_loss = 0.
        seg_out = Variable(torch.FloatTensor())
        if len(M_real):
            if args.detach:
                seg_btoa = btoa.detach()
            else:
                seg_btoa = btoa
            seg_inp = torch.cat((B_real, seg_btoa), dim=1)[indices]
            seg_out = model['seg'](seg_inp)
            for i in range(len(loss_functions)):
                seg_loss += loss_functions[i](seg_out, M_real)
            seg_loss /= len(loss_functions) # average
        g_tot_loss += seg_loss
        g_tot_loss.backward()
        # Update discriminator.
        optimizer['d_a'].zero_grad()
        optimizer['d_b'].zero_grad()
        d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        d_a_loss.backward()
        d_b_loss.backward()
        # Update all networks at once.
        for key in optimizer:
            optimizer[key].step()
        this_metrics = {
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        }
        if len(M_real):
            with torch.no_grad():
                this_metrics.update(metrics['train'](seg_out, M_real))
        return (seg_loss if seg_loss==0 else seg_loss.item(),
                batch,
                (B_real, M_real,
                 seg_out.detach(), btoa.detach(),
                 btoa_atob.detach(), indices),
                this_metrics)

    trainer = Trainer(training_function)

    def validation_function(engine, batch):
        A_real, B_real, M_real, indices = prepare_batch(batch)
        for model_ in model.values():
            model_.eval()
        with torch.no_grad():
            hh, ww = A_real.shape[-2], A_real.shape[-1]
            atob = model['g_atob'](A_real)[:, :, 0:hh, 0:ww]
            atob_btoa = model['g_btoa'](atob)[:, :, 0:hh, 0:ww]
            atob_gen_loss, cycle_aba = compute_g_losses_aba(
                A_real, atob, atob_btoa)
            hh, ww = B_real.shape[-2], B_real.shape[-1]
            btoa = model['g_btoa'](B_real)[:, :, 0:hh, 0:ww]
            btoa_atob = model['g_atob'](btoa)[:, :, 0:hh, 0:ww]
            btoa_gen_loss, cycle_bab = compute_g_losses_bab(
                B_real, btoa, btoa_atob)
            seg_loss = 0.
            seg_out = Variable(torch.FloatTensor())
            if len(M_real):
                seg_inp = torch.cat((B_real, btoa), dim=1)[indices]
                seg_out = model['seg'](seg_inp)
                for i in range(len(loss_functions)):
                    seg_loss += loss_functions[i](seg_out, M_real)
                seg_loss /= len(loss_functions) # average
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, B_real, btoa)
        this_metrics = OrderedDict({
            'atob_gen_loss': atob_gen_loss.data.item(),
            'cycle_aba': cycle_aba.data.item(),
            'btoa_gen_loss': btoa_gen_loss.data.item(),
            'cycle_bab': cycle_bab.data.item(),
            'd_a_loss': d_a_loss.data.item(),
            'd_b_loss': d_b_loss.data.item()
        })
        if len(M_real):
            this_metrics.update(metrics['valid'](seg_out, M_real))
        return (seg_loss if seg_loss==0 else seg_loss.item(),
                batch,
                (B_real, M_real,
                 seg_out.detach(), btoa.detach(),
                 btoa_atob.detach(), indices),
                this_metrics)
    evaluator = Evaluator(validation_function)

    '''
    Reset global Dice score counts every epoch (or validation run).
    '''
    for l in ['1', '2', '4', '124']:
        func = lambda key : \
            metrics[key].measure_functions['dice{}'.format(l)].reset_counts
        trainer.add_event_handler(Events.EPOCH_STARTED, func('train'))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, func('valid'))
    
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

    if not args.no_valid:
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  lambda _: evaluator.run(loader_valid))
        evaluator.add_event_handler(Events.ITERATION_COMPLETED,
                                    progress_valid)
        evaluator.add_event_handler(Events.ITERATION_COMPLETED,
                                    image_saver_valid)

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
            't_module_as_str': t_module_as_str,
            's_module_as_str': s_module_as_str,
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
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)
