from __future__ import (print_function,
                        division)
from builtins import input
from collections import OrderedDict
import sys
import argparse
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
parser = argparse.ArgumentParser(
    description='Minimal-ish example code for doing segmentation on BRATS17.')
parser.add_argument('--arch', type=str, default='fcn_maker')
parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
parser.add_argument('--batch_size', type=int, default=80)
args = parser.parse_args()
assert args.arch in ['vanilla_dilated_fcn', 'fcn_maker']

'''
Settings.
'''
if args.arch == 'vanilla_dilated_fcn':
    model_kwargs = {'in_channels':4, 'C':24, 'classes':3}
else:
    model_kwargs = OrderedDict((
       ('in_channels', 4),
       ('num_classes', 3),
       ('num_init_blocks', 1), # 2
       ('num_main_blocks', 2), # 3
       ('main_block_depth', 1),
       ('init_num_filters', 32),
       ('dropout', 0.05),
       ('main_block', basic_block),
       ('init_block', tiny_block),
       ('norm_kwargs', {'momentum': 0.1}),
       ('nonlinearity', 'ReLU'),
       ('ndim', 2)
    ))
batch_size = args.batch_size

'''
Set paths.
'''
hgg_path = "%s/hgg.h5" % args.data_dir
lgg_path = "%s/lgg.h5" % args.data_dir
model_dir = "models"

if __name__ == '__main__':
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(hgg_path, lgg_path)
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
                                     batch_size=batch_size,
                                     preprocessor=preprocessor_train,
                                     rng=np.random.RandomState(42))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=batch_size,
                                     #nb_io_workers=2,
                                     #nb_proc_workers=2,
                                     preprocessor=preprocessor_valid,
                                     rng=np.random.RandomState(42))

    '''
    Prepare model.
    '''
    torch.backends.cudnn.benchmark = True
    if args.arch == 'vanilla_dilated_fcn':
        model = DilatedFCN(**model_kwargs)
    else:
        model = assemble_resunet(**model_kwargs)
    print(model)
    print("Number of params: %i" % count_params(model))
    model.cuda()
    optimizer = torch.optim.RMSprop(params=model.parameters(),
                                    lr=0.001, alpha=0.9,
                                    weight_decay=1e-4)

    '''
    Set up loss functions and metrics. Since this is
    a multiclass problem, set up a metrics handler for
    each output map.
    '''
    labels = [1,2,4] # 3 classes
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
    Set up logging to screen.
    '''
    def epoch_length(dataset):
        return len(dataset)//batch_size + int(len(dataset)%batch_size>0)
    progress_train = progress_report(epoch_length=epoch_length(data['train']['s']))
    progress_valid = progress_report(epoch_length=epoch_length(data['valid']['s']),
                                     prefix="val",
                                     progress_bar=True)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              Evaluate(evaluator, loader_valid,
                                       epoch_interval=1))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)

    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=20)

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
