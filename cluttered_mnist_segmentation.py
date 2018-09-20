from __future__ import (print_function,
                        division)
from functools import partial
from collections import OrderedDict
import sys
import os
import shutil
import argparse
import warnings

import numpy as np
import torch
from torch.autograd import Variable
import ignite
from ignite.engine import (Events,
                           Engine)
from ignite.handlers import ModelCheckpoint

from data_tools.io import data_flow
from data_tools.data_augmentation import image_random_transform

from utils.common import (experiment,
                          image_logger,
                          scoring_function,
                          summary_tracker)
from utils.metrics import (batchwise_loss_accumulator,
                           dice_loss)
from utils.data import (setup_mnist_data,
                        mnist_data_train,
                        mnist_data_valid,
                        mnist_data_test)
from model import configs
from model.bidomain_segmentation import segmentation_model as model_gan
from model.ae_segmentation import segmentation_model as model_ae

import itertools


'''
Process arguments.
'''
def get_parser():
    parser = argparse.ArgumentParser(description="Cluttered MNIST seg.")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--data_dir', type=str, default='./data/mnist')
    parser.add_argument('--save_path', type=str, default='./experiments')
    mutex_parser = parser.add_mutually_exclusive_group()
    mutex_parser.add_argument('--model_from', type=str, default=None)
    mutex_parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--labeled_fraction', type=float, default=0.1)
    parser.add_argument('--unlabeled_digits', type=int, default=None,
                        nargs='+')
    parser.add_argument('--yield_only_labeled', action='store_true')
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--batch_size_train', type=int, default=20)
    parser.add_argument('--batch_size_valid', type=int, default=20)
    parser.add_argument('--epoch_length', type=int, default=None,
                        help="By default, the training set is pregenerated. "
                             "Otherwise, `epoch_length` batches are "
                             "generated online per epoch.")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='amsgrad',
                        choices=['adam', 'amsgrad', 'rmsprop', 'sgd'])
    parser.add_argument('--grad_penalty', type=float, default=None)
    parser.add_argument('--disc_clip_norm', type=float, default=None)
    parser.add_argument('--n_clutter', type=int, default=8)
    parser.add_argument('--size_clutter', type=int, default=10)
    parser.add_argument('--size_output', type=int, default=100)
    parser.add_argument('--background_noise', type=float, default=0.01)
    parser.add_argument('--pregenerate_training_set', action='store_true')
    parser.add_argument('--n_valid', type=int, default=500)
    parser.add_argument('--n_vis', type=int, default=20)
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--nb_io_workers', type=int, default=1)
    parser.add_argument('--nb_proc_workers', type=int, default=1)
    parser.add_argument('--rseed', type=int, default=1234)
    return parser


if __name__ == '__main__':
    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = True
    
    # Set up experiment.
    experiment_state = experiment(name="mnist", parser=get_parser())
    args = experiment_state.args
    assert args.labeled_fraction > 0
    torch.manual_seed(args.rseed)
    
    # Data augmentation settings.
    da_kwargs = {'rotation_range': 3.,
                 'zoom_range': 0.1,
                 'horizontal_flip': True,
                 'vertical_flip': True,
                 'fill_mode': 'reflect',
                 'spline_warp': True,
                 'warp_sigma': 5,
                 'warp_grid_size': 3}
    
    # Data preprocessing (including data augmentation).
    def preprocessor(warp=False):
        def f(batch):
            s, h, m, _ = zip(*batch[0])
            s = np.expand_dims(s, 1)
            h = np.expand_dims(h, 1)
            m = [np.expand_dims(x, 0) if x is not None else None for x in m]
            if warp:
                for i, (s_, h_, m_) in enumerate(zip(s, h, m)):
                    sm_ = image_random_transform(s_, m_, **da_kwargs)
                    h_  = image_random_transform(h_, **da_kwargs)
                    if m_ is None:
                        sm_ = (sm_, None)
                    s[i], m[i] = sm_
                    h[i]       = h_
            return s, h, m
        return f
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch):
        s, h, m = batch
        # Identify indices of examples with masks.
        indices = [i for i, mask in enumerate(m) if mask is not None]
        m       = [m[i] for i in indices]
        # Prepare for pytorch.
        s = Variable(torch.from_numpy(s))
        h = Variable(torch.from_numpy(h))
        m = Variable(torch.from_numpy(np.array(m)))
        if not args.cpu:
            s = s.cuda()
            h = h.cuda()
            m = m.cuda()
        return s, h, m, indices
    
    # Prepare data.
    data = setup_mnist_data(
        data_dir='./data/mnist',
        n_valid=args.n_valid,
        n_clutter=args.n_clutter,
        size_clutter=args.size_clutter,
        size_output=args.size_output,
        segment_fraction=args.labeled_fraction,
        unlabeled_digits=args.unlabeled_digits,
        yield_only_labeled=args.yield_only_labeled,
        gen_train_online=args.epoch_length is not None,
        background_noise=args.background_noise,
        verbose=True,
        rng=np.random.RandomState(args.rseed))
    n_samples_train = None if args.epoch_length is None \
                           else args.epoch_length*args.batch_size_train
    loader = {
        'train': data_flow([mnist_data_train(data,
                                             length=n_samples_train)],
                            batch_size=args.batch_size_train,
                            sample_random=True,
                            preprocessor=preprocessor(warp=args.augment_data),
                            rng=np.random.RandomState(args.rseed),
                            nb_io_workers=args.nb_io_workers,
                            nb_proc_workers=args.nb_proc_workers),
        'valid': data_flow([mnist_data_valid(data)],
                            batch_size=args.batch_size_valid,
                            preprocessor=preprocessor(warp=False),
                            nb_io_workers=args.nb_io_workers,
                            nb_proc_workers=args.nb_proc_workers),
        'test':  data_flow([mnist_data_test(data)],
                            batch_size=args.batch_size_valid,
                            preprocessor=preprocessor(warp=False),
                            nb_io_workers=args.nb_io_workers,
                            nb_proc_workers=args.nb_proc_workers)}
    
    # Helper for training/validation loops : detach variables from graph.
    def detach(x):
        detached = dict([(k, v.detach())
                         if isinstance(v, Variable)
                         else (k, v)
                         for k, v in x.items()])
        return detached
    
    # Training loop.
    eval_kwargs = {'grad_penalty'  : args.grad_penalty,
                   'disc_clip_norm': args.disc_clip_norm}
    def training_function(engine, batch):
        for model in experiment_state.model.values():
            model.train()
        A, B, M, indices = prepare_batch(batch)
        outputs = experiment_state.model['G'].evaluate(A, B, M, indices,
                                                       **eval_kwargs,
                                         optimizer=experiment_state.optimizer)
        outputs = detach(outputs)
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        A, B, M, indices = prepare_batch(batch)
        outputs = OrderedDict(zip(['out_A', 'out_B', 'out_M'], [A, B, M]))
        with torch.no_grad():
            _outputs = experiment_state.model['G'].evaluate(A, B, M, indices,
                                                            rng=engine.rng)
        _outputs = detach(_outputs)
        outputs.update(_outputs)
        return outputs
    
    # Get engines.
    engines = {}
    append = bool(args.resume_from is not None)
    engines['train'] = experiment_state.setup_engine(
                                            training_function,
                                            append=append,
                                            epoch_length=len(loader['train']))
    engines['valid'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='val',
                                            append=append,
                                            epoch_length=len(loader['valid']))
    engines['test'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='test',
                                            append=append,
                                            epoch_length=len(loader['test']))
    for key in ['valid', 'test']:
        engines[key].add_event_handler(
            Events.STARTED,
            lambda engine: setattr(engine, 'rng', np.random.RandomState(0)))
    
    
    # Set up metrics.
    metrics = {}
    for key in engines:
        metrics[key] = {}
        metrics[key]['dice'] = dice_loss(target_class=1,
                        output_transform=lambda x: (x['out_seg'], x['mask']))
        if isinstance(experiment_state.model, model_ae):
            metrics[key]['loss'] = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_all'])
            metrics[key]['rec']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_rec'])
        if isinstance(experiment_state.model, model_gan):
            metrics[key]['G']    = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_G'])
            metrics[key]['DA']   = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_DA'])
            metrics[key]['DB']   = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_DB'])
            metrics[key]['miA']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_mi_A'])
            metrics[key]['miBA'] = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_mi_BA'])
        for name, m in metrics[key].items():
            m.attach(engines[key], name=name)

    # Set up validation.
    engines['train'].add_event_handler(Events.EPOCH_COMPLETED,
                             lambda _: engines['valid'].run(loader['valid']))
    
    # Set up model checkpointing.
    score_function = scoring_function('dice')
    experiment_state.setup_checkpoints(engines['train'], engines['valid'],
                                       score_function=score_function)
    
    # Set up tensorboard logging for losses.
    tracker = summary_tracker(experiment_state.experiment_path,
                              initial_epoch=experiment_state.get_epoch())
    def _tuple(x):
        if isinstance(x, torch.Tensor) and x.dim()>0:
            return (torch.mean(x, dim=0), len(x))
        return (x, 1)
    for key in ['train', 'valid']:
        tracker.attach(
            engine=engines[key],
            prefix=key,
            output_transform=lambda x: dict([(k, _tuple(v))
                                             for k, v in x.items()
                                             if k.startswith('l_')]))
    
    # Set up image logging to tensorboard.
    def _p(val): return np.squeeze(val.cpu().numpy(), 1)
    save_image = image_logger(
        initial_epoch=experiment_state.get_epoch(),
        directory=os.path.join(experiment_state.experiment_path, "images"),
        summary_tracker=tracker,
        num_vis=min(args.n_vis, args.n_valid),
        output_transform=lambda x: OrderedDict([(k.replace('out_',''), _p(v))
                                                for k, v in x.items()
                                                if k.startswith('out_')
                                                and v is not None]))
    save_image.attach(engines['valid'], name='save_image')
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)
    
    '''
    Test.
    '''
    print("\nTESTING\n")
    engines['test'].run(loader['test'])
