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
import json
import torch
from torch.autograd import Variable
import ignite
from ignite.engine import (Events,
                           Engine)
from ignite.handlers import ModelCheckpoint

from data_tools.io import data_flow
from data_tools.data_augmentation import image_random_transform

from utils.experiment import experiment
from utils.trackers import(image_logger,
                           scoring_function,
                           summary_tracker)
from utils.metrics import (batchwise_loss_accumulator,
                           dice_loss)
from utils.data.common import data_flow_sampler
from utils.data.brats import (prepare_data_brats13s,
                              prepare_data_brats17,
                              preprocessor_brats)
from model import configs
from model.bidomain_segmentation import segmentation_model as model_gan
from model.ae_segmentation import segmentation_model as model_ae

import itertools


'''
Process arguments.
'''
def get_parser():
    parser = argparse.ArgumentParser(description="BRATS seg.")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--dataset', type=str, default='brats13s',
                        choices=['brats13s', 'brats17'])
    parser.add_argument('--data_dir', type=str, default='./data/brats/2013')
    parser.add_argument('--classes', type=str, default='1,2,4',
                        help='Comma-separated list of class labels')
    parser.add_argument('--orientation', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='./experiments')
    mutex_parser = parser.add_mutually_exclusive_group()
    mutex_parser.add_argument('--model_from', type=str, default=None)
    mutex_parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--labeled_fraction', type=float, default=0.1)
    parser.add_argument('--yield_only_labeled', action='store_true')
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--batch_size_train', type=int, default=20)
    parser.add_argument('--batch_size_valid', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=json.loads, default=0.001)
    parser.add_argument('--opt_kwargs', type=json.loads, default=None)
    parser.add_argument('--optimizer', type=str, default='amsgrad')
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
    experiment_state = experiment(name="brats", parser=get_parser())
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
    if not args.augment_data:
        da_kwargs=None
    
    # Prepare data.
    if args.dataset=='brats17':
        prepare_data_brats = prepare_data_brats17
        target_class = [1,2,4]
    elif args.dataset=='brats13s':
        prepare_data_brats = prepare_data_brats13s
        target_class = [4,5]
    else:
        raise ValueError("`dataset` must only be 'brats17' or 'brats13s'")
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"),
                              orientations=args.orientation,
                              masked_fraction=1.-args.labeled_fraction,
                              drop_masked=args.yield_only_labeled,
                              rng=np.random.RandomState(args.rseed))
    data_train = [data['train']['h'], data['train']['s'], data['train']['m']]
    data_valid = [data['valid']['h'], data['valid']['s'], data['valid']['m']]
    loader = {
        'train': data_flow_sampler(data_train,
                                   sample_random=True,
                                   batch_size=args.batch_size_train,
                                   preprocessor=preprocessor_brats(
                                       data_augmentation_kwargs=da_kwargs),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.rseed)),
        'valid': data_flow_sampler(data_valid,
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_brats(
                                       data_augmentation_kwargs=None),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.rseed))}
    
    # Data preprocessing (including data augmentation).
    def preprocessor(warp=False):
        def f(batch):
            h, s, m, _ = zip(*batch[0])
            h = np.expand_dims(h, 1)
            s = np.expand_dims(s, 1)
            m = [np.expand_dims(x, 0) if x is not None else None for x in m]
            if warp:
                for i, (h_, s_, m_) in enumerate(zip(h, s, m)):
                    h_  = image_random_transform(h_, **da_kwargs)
                    sm_ = image_random_transform(s_, m_, **da_kwargs)
                    if m_ is None:
                        sm_ = (sm_, None)
                    h[i]       = h_
                    s[i], m[i] = sm_
            return h, s, m
        return f
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch):
        h, s, m = batch
        # Identify indices of examples with masks.
        indices = [i for i, mask in enumerate(m) if mask is not None]
        m       = [m[i] for i in indices]
        # Remove all but the classes of interest (for visualization).
        m = np.array(m)
        m_filtered = np.zeros_like(m)
        for i in target_class:
            m_filtered[m==i] = i
        # Prepare for pytorch.
        h = Variable(torch.from_numpy(np.array(h)))
        s = Variable(torch.from_numpy(np.array(s)))
        m = Variable(torch.from_numpy(m_filtered))
        if not args.cpu:
            h = h.cuda()
            s = s.cuda()
            m = m.cuda()
        return h, s, m, indices
    
    # Helper for training/validation loops : detach variables from graph.
    def detach(x):
        detached = dict([(k, v.detach())
                         if isinstance(v, Variable)
                         else (k, v)
                         for k, v in x.items()])
        return detached
    
    # Training loop.
    def training_function(engine, batch):
        for model in experiment_state.model.values():
            model.train()
        B, A, M, indices = prepare_batch(batch)
        outputs = experiment_state.model['G'].evaluate(A, B, M, indices,
                                         optimizer=experiment_state.optimizer)
        outputs = detach(outputs)
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        B, A, M, indices = prepare_batch(batch)
        outputs = OrderedDict(zip(['out_B', 'out_A', 'out_M'], [B, A, M]))
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
    engines['valid'].add_event_handler(
        Events.STARTED,
        lambda engine: setattr(engine, 'rng', np.random.RandomState(0)))
    
    
    # Set up metrics.
    metrics = {}
    for key in engines:
        metrics[key] = {}
        metrics[key]['dice'] = dice_loss(target_class=target_class,
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
                                             if k.startswith('l_')
                                             or k.startswith('prob')]),
            metric_keys=['dice'])
    
    # Set up image logging to tensorboard.
    for channel, sequence_name in enumerate(['flair', 't1', 't1c', 't2']):
        def output_transform(output, channel=channel):
            transformed = OrderedDict()
            for k, v in output.items():
                if k.startswith('out_') and v is not None:
                    k_new = k.replace('out_','')
                    v_new = v.cpu().numpy()
                    if k_new in ['M', 'AM']:
                        v_new = np.squeeze(v_new, 1)    # Single channel seg.
                    else:
                        v_new = v_new[:,channel]        # 4 sequences per img.
                    transformed[k_new] = v_new
            return transformed
        save_image = image_logger(
            initial_epoch=experiment_state.get_epoch(),
            directory=os.path.join(experiment_state.experiment_path, "images"),
            summary_tracker=tracker,
            num_vis=args.n_vis,
            suffix=sequence_name,
            output_name=sequence_name,
            output_transform=output_transform,
            fontsize=40)
        save_image.attach(engines['valid'])
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)
    