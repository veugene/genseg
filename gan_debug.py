from __future__ import (print_function,
                        division)
from functools import partial
from collections import OrderedDict
import sys
import os
import shutil
import argparse
from datetime import datetime
import warnings

import numpy as np
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import ignite
from ignite.engine import (Events,
                           Engine)
from ignite.handlers import ModelCheckpoint

from data_tools.io import data_flow
from data_tools.data_augmentation import image_random_transform

from utils.common import (experiment,
                          image_saver,
                          scoring_function,
                          summary_tracker)
from utils.metrics import (batchwise_loss_accumulator,
                           dice_loss)
from utils.data import (setup_mnist_data,
                        mnist_data_train,
                        mnist_data_valid,
                        mnist_data_test)
from model import configs
from model.gan_debug import gan

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
    mutex_parser.add_argument('--model_from', type=str,
                              default="./model/configs/"
                                      "bidomain_segmentation_001.py")
    mutex_parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
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
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--nb_io_workers', type=int, default=1)
    parser.add_argument('--nb_proc_workers', type=int, default=1)
    parser.add_argument('--rseed', type=int, default=1234)
    return parser


if __name__ == '__main__':
    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = False
    
    # Set up experiment.
    experiment_state = experiment(name="mnist", parser=get_parser())
    args = experiment_state.args
    if not args.cpu:
        experiment_state.model.cuda()
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
        s = Variable(torch.from_numpy(s))
        if not args.cpu:
            s = s.cuda()
        return s
    
    # Prepare data.
    data = setup_mnist_data(
        data_dir='./data/mnist',
        n_valid=0,
        n_clutter=args.n_clutter,
        size_clutter=args.size_clutter,
        size_output=args.size_output,
        gen_train_online=args.epoch_length is not None,
        background_noise=args.background_noise,
        verbose=True,
        rng=np.random.RandomState(args.rseed))
    n_samples_train = None if args.epoch_length is None \
                           else args.epoch_length*args.batch_size
    loader = {
        'train': data_flow([mnist_data_train(data,
                                             length=n_samples_train)],
                            batch_size=args.batch_size,
                            sample_random=True,
                            preprocessor=preprocessor(warp=args.augment_data),
                            rng=np.random.RandomState(args.rseed),
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
        experiment_state.model.train()
        experiment_state.optimizer.zero_grad()
        A = prepare_batch(batch)
        outputs = experiment_state.model.evaluate(A,
                                                  **eval_kwargs,
                                                  compute_grad=True)
        experiment_state.optimizer.step()
        outputs = detach(outputs)
        return outputs
    
    # Get engines.
    engines = {}
    append = bool(args.resume_from is not None)
    engines['train'] = experiment_state.setup_engine(
                                            training_function,
                                            append=append,
                                            epoch_length=len(loader['train']))
    
    # Set up image generation for evaluation.
    zc = experiment_state.model._z_sample(20*20)
    zr = experiment_state.model._z_sample(20*20)
    zu = experiment_state.model._z_sample(20*20)
    def gen_image():
        with torch.no_grad():
            out = experiment_state.model.decode(zc, zr, zu)
        out = out.detach().cpu().numpy()
        out = np.transpose(out.reshape(20, 20, out.shape[-2], out.shape[-1]),
                           (0,2,1,3)).reshape(1, 20*out.shape[-2],
                                                 20*out.shape[-1])
        return (out,)
    engines['train'].add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: setattr(engine.state, 'save_images', gen_image()))
    
    # Set up metrics.
    metrics = {}
    for key in engines:
        metrics[key] = {}
        metrics[key]['G'] = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_G'])
        metrics[key]['D'] = batchwise_loss_accumulator(
                        output_transform=lambda x: x['l_D'])
        for name, m in metrics[key].items():
            m.attach(engines[key], name=name)

    # Set up image saving.
    image_saver_obj = image_saver(
        save_path=os.path.join(experiment_state.experiment_path,
                               "generator_outputs"),
        name_images='save_images',
        score_function=None,
        subdirs=False)
    image_saver_obj._call_counter = experiment_state._epoch[0] # For resuming.
    engines['train'].add_event_handler(Events.EPOCH_COMPLETED, 
                                       image_saver_obj)
    
    # Set up model checkpointing.
    experiment_state.setup_checkpoints(engines['train'])
    
    # Set up tensorboard logging.
    # -> Log everything except 'mask' and model outputs.
    trackers = {}
    def _tuple(x):
        if isinstance(x, torch.Tensor) and x.dim()>0:
            return (torch.mean(x, dim=0), len(x))
        return (x, 1)
    for key in engines:
        trackers[key] = summary_tracker(
            path=experiment_state.experiment_path,
            output_transform=lambda x: dict([(k, _tuple(v))
                                             for k, v in x.items()
                                             if k.startswith('l_')]))
        trackers[key].attach(engines[key], prefix=key)
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)
    
