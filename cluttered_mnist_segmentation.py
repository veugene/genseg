from __future__ import (print_function,
                        division)
import argparse
import json
import psutil
from utils.dispatch import (dispatch,
                            dispatch_argument_parser)


'''
Process arguments.
'''
def get_parser():
    parser = dispatch_argument_parser(description="Cluttered MNIST seg.")
    g_exp = parser.add_argument_group('Experiment')
    g_exp.add_argument('--data', type=str, default='./data/mnist')
    g_exp.add_argument('--path', type=str, default='./experiments')
    g_exp.add_argument('--model_from', type=str, default=None)
    g_exp.add_argument('--weights_from', type=str, default=None)
    g_exp.add_argument('--remove_spectral_norm', action='store_true')
    g_exp.add_argument('--weight_decay', type=float, default=1e-4)
    g_exp.add_argument('--labeled_fraction', type=float, default=0.1)
    g_exp.add_argument('--unlabeled_digits', type=int, default=None,
                       nargs='+')
    g_exp.add_argument('--yield_only_labeled', action='store_true')
    g_exp.add_argument('--augment_data', action='store_true')
    g_exp.add_argument('--batch_size_train', type=int, default=20)
    g_exp.add_argument('--batch_size_valid', type=int, default=20)
    g_exp.add_argument('--epoch_length', type=int, default=None,
                       help="By default, the training set is pregenerated. "
                            "Otherwise, `epoch_length` batches are "
                            "generated online per epoch.")
    g_exp.add_argument('--epochs', type=int, default=200)
    g_exp.add_argument('--learning_rate', type=json.loads, default=0.001)
    g_exp.add_argument('--opt_kwargs', type=json.loads, default=None)
    g_exp.add_argument('--optimizer', type=str, default='amsgrad')
    g_exp.add_argument('--n_clutter', type=int, default=8)
    g_exp.add_argument('--size_clutter', type=int, default=10)
    g_exp.add_argument('--size_output', type=int, default=100)
    g_exp.add_argument('--background_noise', type=float, default=0.01)
    g_exp.add_argument('--n_valid', type=int, default=500)
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
    import os

    import numpy as np
    import torch
    from torch.autograd import Variable
    import ignite
    from ignite.engine import (Events,
                               Engine)
    from ignite.handlers import ModelCheckpoint

    from data_tools.io import data_flow
    from data_tools.data_augmentation import image_random_transform

    from utils.data.cluttered_mnist import (setup_mnist_data,
                                            mnist_data_train,
                                            mnist_data_valid,
                                            mnist_data_test)
    from utils.experiment import experiment
    from utils.metrics import (batchwise_loss_accumulator,
                               dice_global)
    from utils.trackers import(image_logger,
                               scoring_function,
                               summary_tracker)
    from model import configs
    from model.ae_segmentation import segmentation_model as model_ae
    from model.common.network.basic import recursive_remove_spectral_norm

    import itertools


    # Disable buggy profiler.
    torch.backends.cudnn.benchmark = True
    
    # Set up experiment.
    experiment_state = experiment(args)
    if args.remove_spectral_norm:
        for name, submodel in experiment_state.model.items():
            print("DEBUG removing spectral norm:", name)
            recursive_remove_spectral_norm(submodel)
    args = experiment_state.args
    assert args.labeled_fraction > 0
    torch.manual_seed(args.init_seed)
    
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
        # Prepare for pytorch.
        h = Variable(torch.from_numpy(h)).cuda()
        s = Variable(torch.from_numpy(s)).cuda()
        return h, s, m
    
    # Prepare data.
    data = setup_mnist_data(
        data_dir=args.data,
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
        rng=np.random.RandomState(args.data_seed))
    n_samples_train = None if args.epoch_length is None \
                           else args.epoch_length*args.batch_size_train
    loader = {
        'train': data_flow([mnist_data_train(data,
                                             length=n_samples_train)],
                            batch_size=args.batch_size_train,
                            sample_random=True,
                            preprocessor=preprocessor(warp=args.augment_data),
                            rng=np.random.RandomState(args.init_seed),
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
        return outputs
    
    # Validation loop.
    def validation_function(engine, batch):
        for model in experiment_state.model.values():
            model.eval()
        B, A, M = prepare_batch(batch)
        outputs = OrderedDict(zip(['x_A', 'x_B'], [A, B]))
        with torch.no_grad():
            _outputs = experiment_state.model['G'](A, B, M, rng=engine.rng)
        _outputs = detach(_outputs)
        outputs.update(_outputs)
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
        metrics[key] = {}
        metrics[key]['dice'] = dice_global(target_class=1,
                        output_transform=lambda x: (x['x_AM'], x['x_M']))
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
        num_vis=min(args.n_vis, args.n_valid),
        output_transform=lambda x: OrderedDict([(k.replace('x_',''), _p(v))
                                                for k, v in x.items()
                                                if k.startswith('x_')
                                                and v is not None]),
        fontsize=12)
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