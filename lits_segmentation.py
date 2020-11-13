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
    parser = dispatch_argument_parser(description="LiTS seg.")
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--data', type=str, default='./data/lits/lits.h5')
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
    g_exp.add_argument('--model_kwargs', type=json.loads, default=None)
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
    g_exp.add_argument('--num_workers', type=int, default=2)
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
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    import ignite
    from ignite.engine import (Events,
                               Engine)
    from ignite.handlers import ModelCheckpoint

    from utils.data.lits import (prepare_data_lits,
                                 collate_lits)
    from utils.experiment import experiment
    from utils.metrics import (batchwise_loss_accumulator,
                               dice_global)
    from utils.trackers import(image_logger,
                               scoring_function,
                               summary_tracker)
    from model import configs
    from model.ae_segmentation import segmentation_model as model_ae
    from model.bd_segmentation import segmentation_model as model_bd
    from model.mean_teacher_segmentation import segmentation_model as model_mt
    
    
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
    dataset = prepare_data_lits(path=args.data,
                                masked_fraction=1.-args.labeled_fraction,
                                drop_masked=args.yield_only_labeled,
                                rng=np.random.RandomState(args.data_seed))
    loader = {
        'train': DataLoader(dataset['train'],
                            batch_size=args.batch_size_train,
                            shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=collate_lits),
        'valid': DataLoader(dataset['valid'],
                            batch_size=args.batch_size_valid,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_lits),
        'test':  DataLoader(dataset['test'],
                            batch_size=args.batch_size_valid,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_lits)}
    
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
        B, A, M = batch
        B, A = B.cuda(), A.cuda()
        outputs = experiment_state.model['G'](A, B, M,
                                         optimizer=experiment_state.optimizer)
        outputs = detach(outputs)
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        B, A, M = batch
        B, A = B.cuda(), A.cuda()
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
    engines['test'] = experiment_state.setup_engine(
                                            validation_function,
                                            prefix='test',
                                            epoch_length=len(loader['test']))
    for key in ['valid', 'test']:
        engines[key].add_event_handler(
            Events.STARTED,
            lambda engine: setattr(engine, 'rng', np.random.RandomState(0)))
    
    
    # Set up metrics.
    metrics = {}
    for key in engines:
        metrics[key] = OrderedDict()
        metrics[key]['dice'] = dice_global(target_class=1,
                        output_transform=lambda x: (x['x_AM'], x['x_M']))
        if isinstance(experiment_state.model['G'], model_ae):
            metrics[key]['rec']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_rec'])
            metrics[key]['loss'] = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_all'])
        elif isinstance(experiment_state.model['G'], model_bd):
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
        elif isinstance(experiment_state.model['G'], model_mt):
            metrics[key]['seg']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_seg'])
            metrics[key]['con']  = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_con'])
            metrics[key]['loss'] = batchwise_loss_accumulator(
                            output_transform=lambda x: x['l_all'])
        else:
            pass
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
    def prepare_images(output):
        items = []
        for k, v in output.items():
            if k.startswith('x_') and v is not None:
                stack = v[:,0,:,:].cpu().numpy()
                if k.endswith('M'):
                    # This is a mask. Rescale to to within [-1, 1] for 
                    # visualization.
                    stack = 2*np.clip(mask, 0, 1)-1
                else:
                    stack = np.clip(mask, -1, 1)
                items.append((k.replace('x_', ''), stack))
        return OrderedDict(items)
    save_image = image_logger(
        initial_epoch=experiment_state.get_epoch(),
        directory=os.path.join(experiment_state.experiment_path, "images"),
        summary_tracker=tracker if args.save_image_events else None,
        num_vis=args.n_vis,
        min_val=-1,
        max_val=1,
        output_transform=prepare_images,
        fontsize=48)
    save_image.attach(engines['valid'])
    
    '''
    Train.
    '''
    engines['train'].run(loader['train'], max_epochs=args.epochs)
    
    '''
    Test.
    '''
    print("\nTESTING\n")
    engines['test'].run(loader['test'])
    print("\nTESTING ON BEST CHECKPOINT\n")
    experiment_state.load_best_state()
    engines['test'].run(loader['test'])


if __name__ == '__main__':
    parser = get_parser()
    dispatch(parser, run)