from __future__ import (print_function,
                        division)
import argparse
from collections import OrderedDict
from functools import partial
import os
import re
import shutil
import subprocess
import sys
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

from utils.data.brats import (prepare_data_brats13s,
                              prepare_data_brats17,
                              preprocessor_brats)
from utils.data.common import data_flow_sampler
from utils.dispatch import (dispatch,
                            dispatch_parser)
from utils.experiment import experiment
from utils.metrics import (batchwise_loss_accumulator,
                           dice_global)
from utils.trackers import(image_logger,
                           scoring_function,
                           summary_tracker)

from model import configs
from model.ae_segmentation import segmentation_model as model_ae


'''
Process arguments.
'''
def get_parser():
    parser = argparse.ArgumentParser(description="BRATS seg.",
                                     parents=[dispatch_parser])
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--dataset', type=str, default='brats13s',
                       choices=['brats13s', 'brats17'])
    g_exp.add_argument('--data_dir', type=str, default='./data/brats/2013')
    g_exp.add_argument('--slice_conditional', action='store_true')
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
    g_exp.add_argument('--resume', action='store_true')
    g_exp.add_argument('--weights_from', type=str, default=None)
    g_exp.add_argument('--weight_decay', type=float, default=1e-4)
    g_exp.add_argument('--labeled_fraction', type=float, default=0.1)
    g_exp.add_argument('--yield_only_labeled', action='store_true')
    g_exp.add_argument('--augment_data', action='store_true')
    g_exp.add_argument('--batch_size_train', type=int, default=20)
    g_exp.add_argument('--batch_size_valid', type=int, default=20)
    g_exp.add_argument('--epochs', type=int, default=200)
    g_exp.add_argument('--learning_rate', type=json.loads, default=0.001)
    g_exp.add_argument('--opt_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--optimizer', type=str, default='amsgrad')
    g_exp.add_argument('--n_vis', type=int, default=20)
    g_exp.add_argument('--nb_io_workers', type=int, default=1)
    g_exp.add_argument('--nb_proc_workers', type=int, default=1)
    g_exp.add_argument('--save_image_events', action='store_true',
                       help="Save images into tensorboard event files.")
    g_exp.add_argument('--init_seed', type=int, default=1234)
    g_exp.add_argument('--data_seed', type=int, default=0)
    return parser


def run():
    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = True
    
    # Set up experiment.
    experiment_state = experiment(parser=get_parser())
    args = experiment_state.args
    assert args.labeled_fraction > 0
    torch.manual_seed(args.init_seed)
    
    # Data augmentation settings.
    da_kwargs = {'rotation_range': 3.,
                 'zoom_range': 0.1,
                 'intensity_shift_range': 0.1,
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
                              masked_fraction=1.-args.labeled_fraction,
                              drop_masked=args.yield_only_labeled,
                              rng=np.random.RandomState(args.data_seed))
    data_train = [data['train']['h'], data['train']['s'], data['train']['m'],
                  data['train']['hi'], data['train']['si']]
    data_valid = [data['valid']['h'], data['valid']['s'], data['valid']['m'],
                  data['valid']['hi'], data['valid']['si']]
    loader = {
        'train': data_flow_sampler(data_train,
                                   sample_random=True,
                                   batch_size=args.batch_size_train,
                                   preprocessor=preprocessor_brats(
                                       data_augmentation_kwargs=da_kwargs),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed)),
        'valid': data_flow_sampler(data_valid,
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_brats(
                                       data_augmentation_kwargs=None),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed))}
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch, slice_conditional=False):
        h, s, m, h_indices, s_indices = batch
        # Prepare for pytorch.
        h = Variable(torch.from_numpy(np.array(h))).cuda()
        s = Variable(torch.from_numpy(np.array(s))).cuda()
        # Provide slice index tuple if slice_conditional.
        if not slice_conditional:
            h_indices = s_indices = None
        return h, s, m, h_indices, s_indices
    
    # Helper for training/validation loops : detach variables from graph.
    def detach(x):
        detached = OrderedDict([(k, v.detach())
                                if isinstance(v, Variable)
                                else (k, v)
                                for k, v in x.items()])
        return detached
    
    # Training loop.
    def training_function(engine, batch):
        for model in experiment_state.model.values():
            model.train()
        B, A, M, I_A, I_B = prepare_batch(batch, args.slice_conditional)
        outputs = experiment_state.model['G'](A, B, M,
                                         optimizer=experiment_state.optimizer,
                                         class_A=I_A, class_B=I_B)
        outputs = detach(outputs)
        
        # Drop images without labels, for visualization.
        indices = [i for i, m in enumerate(M) if m is not None]
        for key in filter(lambda x: x.startswith('x_'), outputs.keys()):
            if outputs['x_M'] is None:
                outputs[key] = None
            elif outputs[key] is not None and key not in ['x_M', 'x_AM']:
                outputs[key] = outputs[key][indices]
        
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        B, A, M, I_A, I_B = prepare_batch(batch, args.slice_conditional)
        with torch.no_grad():
            outputs = experiment_state.model['G'](A, B, M, rng=engine.rng,
                                                  class_A=I_A, class_B=I_B)
        outputs = detach(outputs)
        return outputs
    
    # Get engines.
    engines = {}
    engines['train'] = experiment_state.setup_engine(
                                            training_function,
                                            append=args.resume,
                                            epoch_length=len(loader['train']))
    engines['valid'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='val',
                                            append=args.resume,
                                            epoch_length=len(loader['valid']))
    engines['valid'].add_event_handler(
        Events.STARTED,
        lambda engine: setattr(engine, 'rng', np.random.RandomState(0)))
    
    
    # Set up metrics.
    metrics = {}
    def dice_transform_all(x):
        if x['x_AM'] is not None and x['x_AM'].size(1)!=1:
            return (1.-x['x_AM'][:,0:1], x['x_M'])  # 4 out; get whole lesion.
        return (x['x_AM'], x['x_M'])                # 1 out; get whole lesion.
    def dice_transform_single(x):
        if x['x_AM'] is not None and x['x_AM'].size(1)!=1:
            return (x['x_AM'], x['x_M'])            # 4 out; get single class.
        return (None, x['x_M'])                     # 1 out; do nothing. HACK
    for key in engines:
        metrics[key] = OrderedDict()
        metrics[key]['dice'] = dice_global(target_class=target_class,
                                           output_transform=dice_transform_all)
        for i, c in enumerate(target_class):
            metrics[key]['dice{}'.format(c)] = dice_global(
                target_class=c,
                prediction_index=i+1,
                output_transform=dice_transform_single)
        metrics[key]['rec']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_rec'])
        if isinstance(experiment_state.model['G'], model_ae):
            metrics[key]['loss'] = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_all'])
        else:
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
    for key in ['train', 'valid']:
        tracker.attach(
            engine=engines[key],
            prefix=key,
            output_transform=lambda x: dict([(k, v)
                                             for k, v in x.items()
                                             if k.startswith('l_')]),
            metric_keys=['dice']+['dice{}'.format(c) for c in target_class])
    
    # Set up image logging.
    for channel, sequence_name in enumerate(['flair', 't1', 't1c', 't2']):
        def output_transform(output, channel=channel):
            transformed = OrderedDict()
            for k, v in output.items():
                if k.startswith('x_') and v is not None and v.dim()==4:
                    k_new = k.replace('x_','')
                    v_new = v.cpu().numpy()
                    if k_new in ['M', 'AM']:
                        if v_new.shape[1]==1:
                            # 'M', or 'AM' with single class.
                            v_new = np.squeeze(v_new, axis=1)
                        else:
                            # 'AM' with multiple classes.
                            v_new = np.argmax(v_new, axis=1)
                            v_new = sum([(v_new==i+1)*c
                                         for i, c in enumerate(target_class)])
                        if output['x_AM'] is not None and \
                                                    output['x_AM'].shape[1]!=1:
                            # HACK (min and max values for correct vis)
                            v_new[:,0,0]  = max(target_class)
                            v_new[:,0,-1] = 0
                    else:
                        v_new = v_new[:,channel]         # 4 sequences per img.
                    transformed[k_new] = v_new
            return transformed
        for key in ['train', 'valid']:
            save_image = image_logger(
                initial_epoch=experiment_state.get_epoch(),
                directory=os.path.join(experiment_state.experiment_path,
                                    "images/{}".format(key)),
                summary_tracker=(tracker if key=='valid'
                                         and args.save_image_events else None),
                num_vis=args.n_vis,
                suffix=sequence_name,
                output_name=sequence_name,
                output_transform=output_transform,
                fontsize=40)
            save_image.attach(engines[key])
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)


if __name__ == '__main__':
    parser = get_parser()
    dispatch(parser, run)