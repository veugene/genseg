from __future__ import (print_function,
                        division)
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
from ignite.trainer import Trainer
from ignite.engine import Events
from ignite.evaluator import Evaluator
from ignite.handlers import ModelCheckpoint

from architectures.image2image import DilatedFCN
from utils.ignite import (progress_report,
                          metrics_handler,
                          image_saver)
from utils.metrics import (dice_loss,
                           accuracy)
from utils.data import data_flow_sampler
from utils.data import prepare_data_brats
from utils.data import preprocessor_brats
from util import count_params
import configs
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation on BRATS 2017.")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    g_load = parser.add_mutually_exclusive_group(required=False)
    g_load.add_argument('--model_from', type=str, default='resunet')
    g_load.add_argument('--resume', type=str, default=None)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    return args

  
if __name__ == '__main__':
    args = parse_args()

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

    def get_optimizer(name, model, lr):
        if name == 'RMSprop':
            optimizer = torch.optim.RMSprop(params=model.parameters(),
                                            lr=lr,
                                            alpha=0.9,
                                            weight_decay=args.weight_decay)
            return optimizer
        else:
            raise NotImplemented("Optimizer {} not supported."
                                "".format(args.optimizer))

    '''
    Prepare model. The `resume` arg is able to restore the model,
    its state, and the optimizer's state, whereas `model_from`
    (which is mutually exclusive with `resume`) only loads the
    desired architecture.
    '''
    torch.backends.cudnn.benchmark = False   # profiler
    if args.resume is not None:
        saved_dict = torch.load(args.resume)
        model = getattr(configs, saved_dict['model_from'])()
        if args.cuda:
            model.cuda()
        model.load_state_dict(saved_dict['weights'])
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
        #import pdb
        #pdb.set_trace()
        optimizer.load_state_dict(saved_dict['optim'])
    else:
        model = getattr(configs, args.model_from)()
        if args.cuda:
            model.cuda()
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    print("Number of parameters: {}".format(count_params(model)))

    '''
    Set up experiment directory.
    '''
    name = "brats_seg" if args.name is None else args.name
    exp_id = "%s_{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now()) % name
    path = os.path.join(args.save_path, exp_id)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "cmd.sh"), 'w') as f:
        f.write(' '.join(sys.argv))
        
    '''
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
    '''
    labels = [0,1,2,4] # 4 classes
    loss_functions = []
    metrics_dict = OrderedDict()
    for idx,l in enumerate(labels):
        dice = dice_loss(l,idx)
        if args.cuda:
            dice = dice.cuda()
        loss_functions.append(dice)
        metrics_dict['dice{}'.format(l)] = dice
    metrics = metrics_handler(metrics_dict)
    
    '''
    Visualize validation outputs.
    '''
    bs = args.batch_size
    epoch_length = lambda ds : len(ds)//bs + int(len(ds)%bs>0)
    num_batches_valid = epoch_length(data['valid']['s'])
    image_saver_valid = image_saver(save_path=os.path.join(path, "validation"),
                                    epoch_length=num_batches_valid)
    
    '''
    Set up training and evaluation functions.
    '''
    def prepare_batch(batch):
        b0, b1 = batch
        b0 = Variable(torch.from_numpy(np.array(b0)))
        b1 = Variable(torch.from_numpy(np.array(b1)))
        if args.cuda:
            b0 = b0.cuda()
            b1 = b1.cuda()
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
            image_saver_valid(batch[0], batch[1], output)
            return loss.data.item(), metrics(output, batch[1])
    evaluator = Evaluator(validation_function)
    
    '''
    Set up logging to screen.
    '''
    bs = args.batch_size
    progress_train = progress_report(prefix="tr",
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    cpt_handler = ModelCheckpoint(dirname=path,
                                  filename_prefix='weights',
                                  save_interval=1,
                                  n_saved=5,
                                  atomic=False,
                                  exist_ok=True,
                                  create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              cpt_handler,
                              {'model':
                               {'weights': model.state_dict(),
                                'model_from': args.model_from,
                                'optim': optimizer.state_dict()}
                              })
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda engine, state: evaluator.run(loader_valid))
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
