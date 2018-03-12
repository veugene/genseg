from __future__ import (print_function,
                        division)
from collections import OrderedDict
import sys
import os
import shutil
import argparse
from datetime import datetime

import numpy as np
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import ignite
from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from ignite.engine import Events
from ignite.handlers import Evaluate

from architectures.image2image import DilatedFCN
from utils.ignite import (progress_report,
                          metrics_handler)
from utils.metrics import (dice_loss,
                           accuracy)
from utils.data import data_flow_sampler
from utils.data import prepare_data_brats
from utils.data import preprocessor_brats
from util import count_params
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation on BRATS 2017.")
    parser.add_argument('--arch', type=str, default='resunet')
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    g_load = parser.add_mutually_exclusive_group(required=False)
    g_load.add_argument('--model_from', type=str, default=None)
    g_load.add_argument('--load', type=str, default=None)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    args = parser.parse_args()
    return args

'''
Settings.
'''
model_kwargs_dilated = {'in_channels': 4, 'C': 24, 'classes': 3}
model_kwargs = OrderedDict((
    ('in_channels', 4),
    ('num_classes', 4),
    ('num_init_blocks', 2),
    ('num_main_blocks', 3),
    ('main_block_depth', 1),
    ('init_num_filters', 32),
    ('dropout', 0.),
    ('main_block', basic_block),
    ('init_block', tiny_block),
    ('norm_kwargs', {'momentum': 0.1}),
    ('nonlinearity', 'ReLU'),
    ('ndim', 2)
))


if __name__ == '__main__':
    args = parse_args()
    assert args.arch in ['vanilla_dilated_fcn', 'resunet']
    if args.arch=='vanilla_dilated_fcn':
        model_kwargs = model_kwargs_dilated
    
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"))
    data_train = [data['train']['s'], data['train']['m']]
    data_valid = [data['valid']['s'], data['valid']['m']]
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
                                     nb_proc_workers=3,
                                     rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=args.batch_size,
                                     preprocessor=preprocessor_valid,
                                     nb_io_workers=1,
                                     nb_proc_workers=0,
                                     rng=np.random.RandomState(42))

    '''
    Prepare model.
    '''
    if args.model_from is not None:
        raise NotImplemented("Loading model with \'--model_from\' not "
                             "implemented.")
    elif args.load is not None:
        raise NotImplemented("Loading model with \'--load\' not implemented.")
    else:
        torch.backends.cudnn.benchmark = False   # profiler
        if args.arch == 'vanilla_dilated_fcn':
            model = DilatedFCN(**model_kwargs)
            print(model)
        else:
            model = assemble_resunet(**model_kwargs)
        print("Number of parameters: {}".format(count_params(model)))
        model.cuda()
        if args.optimizer=='RMSprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(),
                                            lr=args.learning_rate,
                                            alpha=0.9,
                                            weight_decay=args.weight_decay)
        else:
            raise NotImplemented("Optimizer {} not supported."
                                "".format(args.optimizer))

    '''
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
    '''
    labels = [0,1,2,4] # 4 classes
    loss_functions = []
    metrics_dict = OrderedDict()
    for idx,l in enumerate(labels):
        dice = dice_loss(l,idx).cuda()
        loss_functions.append(dice)
        metrics_dict['dice{}'.format(l)] = dice
    metrics = metrics_handler(metrics_dict)

    '''
    Set up training and evaluation functions.
    '''
    def prepare_batch(batch):
        b0, b1 = batch
        b0 = Variable(torch.from_numpy(np.array(b0))).cuda()
        b1 = Variable(torch.from_numpy(np.array(b1))).cuda()
        return b0, b1

    def training_function(batch):
        batch = prepare_batch(batch)
        model.train()
        optimizer.zero_grad()
        output = model(batch[0])
        loss = 0.
        for i in range(len(loss_functions)):
            loss += loss_functions[i](output, batch[1])
        loss /= len(loss_functions) # average
        loss.backward()
        optimizer.step()
        return loss.data.item(), metrics(output, batch[1])
    trainer = Trainer(training_function)

    def validation_function(batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            output = model(batch[0])
            loss = 0.
            for i in range(len(loss_functions)):
                loss += loss_functions[i](output, batch[1])
            return loss.data.item(), metrics(output, batch[1])
    evaluator = Evaluator(validation_function)

    '''
    Set up experiment directory.
    '''
    exp_id = "brats_seg_{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
    path = os.path.join(args.save_path, exp_id, "cmd.sh")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(' '.join(sys.argv))
    if args.model_from is not None:
        model_fn = os.path.basename(args.model_from)
        shutil.copyfile(args.model_from, os.path.join(path, model_fn))

    '''
    Set up logging to screen.
    '''
    bs = args.batch_size
    epoch_length = lambda ds : len(ds)//bs + int(len(ds)%bs>0)
    num_batches_train = epoch_length(data['train']['s'])
    num_batches_valid = epoch_length(data['valid']['s'])
    progress_train = progress_report(epoch_length=num_batches_train,
                                     logfile=os.path.join(path,
                                                          "log_train.txt"))
    progress_valid = progress_report(epoch_length=num_batches_valid,
                                     prefix="val",
                                     progress_bar=True,
                                     logfile=os.path.join(path,
                                                          "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              Evaluate(evaluator,
                                       loader_valid,
                                       epoch_interval=1))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)

    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=args.epochs)

    '''
    Predict and save images. Right now, this is just
    the first element of the first minibatch.
    '''
    for batch in loader_train:
        X_batch, M_batch = prepare_batch(batch)
        output = model(X_batch)[0] # (k,h,w)
        output_c1 = output[0].data.cpu().numpy()
        output_c2 = output[1].data.cpu().numpy()
        output_c4 = output[2].data.cpu().numpy()
        h, w = output.shape[-2], output.shape[-1]
        # grid for prediction
        grid = np.zeros((h*1, w*3))
        grid[:, 0:w] = output_c1
        grid[:, w:(w*2)] = output_c2
        grid[:, (w*2):(w*3)] = output_c4
        # grid for ground truth
        M = np.asarray(batch[1])[0]
        grid2 = np.zeros((h*1, w*3))
        grid2[:, 0:w] = (M[0] == 1)
        grid2[:, w:(w*2)] = (M[0] == 2)
        grid2[:, (w*2):(w*3)] = (M[0] == 4)
        # combine
        grid_combined = np.vstack((grid, grid2))
        imsave(arr=grid_combined, name="test.png")
        break
