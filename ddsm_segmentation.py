from __future__ import (print_function,
                        division)
import argparse
import json
from utils.dispatch import (dispatch,
                            dispatch_argument_parser)


'''
Process arguments.
'''
def get_parser():
    parser = dispatch_argument_parser(description="DDSM seg.")
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--data', type=str, default='./data/ddsm/ddsm.h5')
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
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


def run(args):
    from collections import OrderedDict
    from functools import partial
    import os
    import re
    import shutil
    import subprocess
    import sys
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

    from utils.data.common import data_flow_sampler
    from utils.data.ddsm import (prepare_data_ddsm,
                                 preprocessor_ddsm)
    from utils.experiment import experiment
    from utils.metrics import (batchwise_loss_accumulator,
                               dice_global)
    from utils.trackers import(image_logger,
                               scoring_function,
                               summary_tracker)
    from model import configs
    from model.ae_segmentation import segmentation_model as model_ae
    
    
    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = True
    
    # Set up experiment.
    experiment_state = experiment(args)
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
    data = prepare_data_ddsm(path=args.data,
                             masked_fraction=1.-args.labeled_fraction,
                             drop_masked=args.yield_only_labeled,
                             rng=np.random.RandomState(args.data_seed))
    data_train = [data['train']['h'], data['train']['s'], data['train']['m']]
    data_valid = [data['valid']['h'], data['valid']['s'], data['valid']['m']]
    loader = {
        'train': data_flow_sampler(data_train,
                                   sample_random=True,
                                   batch_size=args.batch_size_train,
                                   preprocessor=preprocessor_ddsm(
                                       data_augmentation_kwargs=da_kwargs),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed)),
        'valid': data_flow_sampler(data_valid,
                                   sample_random=True,
                                   batch_size=args.batch_size_valid,
                                   preprocessor=preprocessor_ddsm(
                                       data_augmentation_kwargs=None),
                                   nb_io_workers=args.nb_io_workers,
                                   nb_proc_workers=args.nb_proc_workers,
                                   rng=np.random.RandomState(args.init_seed))}
    
    # Function to convert data to pytorch usable form.
    def prepare_batch(batch):
        h, s, m = batch
        # Prepare for pytorch.
        h = Variable(torch.from_numpy(np.array(h))).cuda()
        s = Variable(torch.from_numpy(np.array(s))).cuda()
        return h, s, m
    
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
        B, A, M = prepare_batch(batch)
        outputs = experiment_state.model['G'](A, B, M,
                                         optimizer=experiment_state.optimizer)
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
        B, A, M = prepare_batch(batch)
        with torch.no_grad():
            outputs = experiment_state.model['G'](A, B, M, rng=engine.rng)
        outputs = detach(outputs)
        return outputs
    
    # Get engines.
    engines = {}
    engines['train'] = experiment_state.setup_engine(
                                            training_function,
                                            epoch_length=len(loader['train']))
    engines['valid'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='val',
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
        metrics[key]['dice'] = dice_global(target_class=1,
                                           output_transform=dice_transform_all)
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
            metric_keys=['dice'])
    
    # Set up image logging to tensorboard.
    def _p(val): return np.squeeze(val.cpu().numpy(), 1)
    save_image = image_logger(
        initial_epoch=experiment_state.get_epoch(),
        directory=os.path.join(experiment_state.experiment_path, "images"),
        summary_tracker=tracker if args.save_image_events else None,
        num_vis=args.n_vis,
        output_transform=lambda x: OrderedDict([(k.replace('x_',''), _p(v))
                                                for k, v in x.items()
                                                if k.startswith('x_')
                                                and v is not None]),
        fontsize=48)
    save_image.attach(engines['valid'])
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)


if __name__ == '__main__':
    parser = get_parser()
    dispatch(parser, run)