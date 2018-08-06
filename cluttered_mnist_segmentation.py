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
#import imp

import numpy as np
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import ignite

from ignite.engine import (Events,
                           Engine)
from ignite.handlers import ModelCheckpoint

from utils.common import (experiment,
                          image_saver)
from utils.ignite import (metrics_handler,
                          scoring_function)
from utils.metrics import (dice_loss,
                           accuracy)
from utils.data import (setup_mnist_data,
                        autorewind)
from model import configs

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
    parser.add_argument('--labeled_fraction', type=float, default=0.1)
    parser.add_argument('--yield_only_labeled', action='store_true')
    parser.add_argument('--batch_size_train', type=int, default=20)
    parser.add_argument('--batch_size_valid', type=int, default=20)
    parser.add_argument('--epoch_length', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='amsgrad',
                        choices=['adam', 'amsgrad', 'rmsprop', 'sgd'])
    parser.add_argument('--n_clutter', type=int, default=8)
    parser.add_argument('--size_clutter', type=int, default=10)
    parser.add_argument('--size_output', type=int, default=100)
    parser.add_argument('--n_valid', type=int, default=500)
    parser.add_argument('--cpu', default=False, action='store_true')
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
    assert args.labeled_fraction > 0
    torch.manual_seed(args.rseed)
    
    # Prepare data.
    rng = np.random.RandomState(args.rseed)
    data = setup_mnist_data(
        data_dir='./data/mnist',
        n_valid=args.n_valid,
        n_clutter=args.n_clutter,
        size_clutter=args.size_clutter,
        size_output=args.size_output,
        segment_fraction=args.labeled_fraction,
        yield_only_labeled=args.yield_only_labeled,
        verbose=True,
        rng=rng)
    loader_train = autorewind(partial(data.gen_train,
                                      args.batch_size_train,
                                      args.epoch_length))
    loader_valid = autorewind(partial(data.gen_valid, args.batch_size_valid))
    loader_test  = autorewind(partial(data.gen_test,  args.batch_size_valid))
    
    def prepare_batch(batch, labeled_fraction=1.):
        s, h, m, _ = zip(*batch)
        
        # Identify indices of examples with masks.
        indices = [i for i, mask in enumerate(m) if mask is not None]
        m_      = [m[i] for i in indices]
        
        # Prepare for pytorch.
        s = Variable(torch.from_numpy(np.expand_dims(s, 1)))
        h = Variable(torch.from_numpy(np.expand_dims(h, 1)))
        m = Variable(torch.from_numpy(np.expand_dims(m_, 1)))
        if not args.cpu:
            s = s.cuda()
            h = h.cuda()
            m = m.cuda()
        return s, h, m, indices
    
    # Set up metrics.
    metrics = {'train': None, 'valid': None}
    for key in ['train', 'valid']:
        metrics_dict = OrderedDict((
            ('dice',   dice_loss(1, 0)),
            ('g_dice', dice_loss(1, 0, accumulate=True))
            ))
        metrics[key] = metrics_handler(metrics_dict)
    
    # Training loop.
    def training_function(engine, batch):
        experiment_state.model.train()
        experiment_state.optimizer.zero_grad()
        A, B, M, indices = prepare_batch(batch, args.labeled_fraction)
        losses, outputs = experiment_state.model.evaluate(A, B, M, indices,
                                                          compute_grad=True)
        experiment_state.optimizer.step()
        metrics_dict = {}
        if len(indices)>0:
            with torch.no_grad():
                metrics_dict = metrics['train'](outputs['seg'], M)
        setattr(engine.state, 'metrics', metrics_dict)
        return losses['loss'].item(), losses, metrics_dict
    
    # Validation loop.
    def validation_function(engine, batch):
        experiment_state.model.eval()
        A, B, M, indices = prepare_batch(batch)
        with torch.no_grad():
            losses, outputs = experiment_state.model.evaluate(
                                        A, B, M, indices, compute_grad=False)
            metrics_dict = metrics['train'](outputs['seg'], M)
        setattr(engine.state, 'metrics', metrics_dict)
        
        # Prepare images to save to disk.
        s, h, m, _ = zip(*batch)
        images = (np.array(s), np.array(h), np.array(m))
        images += tuple([np.squeeze(outputs[key].cpu().numpy(), 1)[:len(A)]
                        for key in outputs.keys() if outputs[key] is not None])
        setattr(engine.state, 'save_images', images)
        
        return losses['loss'].item(), losses, metrics_dict
    
    # Get engines.
    append = bool(args.resume_from is not None)
    trainer   = experiment_state.setup_engine(training_function,
                                              append=append,
                                              epoch_length=args.epoch_length)
    epoch_length_val =(     len(data._validation_set)//args.batch_size_valid
                       +int(len(data._validation_set)%args.batch_size_valid>0))
    evaluator = experiment_state.setup_engine(validation_function,
                                              prefix='val',
                                              append=append,
                                              epoch_length=epoch_length_val)
    tester    = experiment_state.setup_engine(validation_function,
                                              prefix='test',
                                              append=append,
                                              epoch_length=epoch_length_val)
    
    # Reset global Dice score counts every epoch (or validation run).
    func = lambda key : metrics[key].measure_functions['g_dice'].reset_counts
    trainer.add_event_handler(Events.EPOCH_STARTED, func('train'))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, func('valid'))

    # Set up validation.
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(loader_valid))
    
    # Set up image saving.
    image_saver_obj = image_saver(
        save_path=os.path.join(experiment_state.experiment_path,
                               "validation_outputs"),
        name_images='save_images',
        score_function=scoring_function('metrics', 'g_dice'))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_obj)
    
    # Set up model checkpointing.
    score_function = scoring_function('metrics', 'g_dice')
    experiment_state.setup_checkpoints(trainer, evaluator,
                                       score_function=score_function)
    
    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)
    
    '''
    Test.
    '''
    print("\nTESTING\n")
    tester.run(loader_test)
