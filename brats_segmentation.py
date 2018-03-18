from __future__ import (print_function,
                        division)
from collections import OrderedDict
import sys
import os
import shutil
import argparse
from datetime import datetime
import warnings
import imp

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
                          scoring_function)
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
    parser.add_argument('--name', type=str, default="brats_seg")
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    g_load = parser.add_mutually_exclusive_group(required=False)
    g_load.add_argument('--model_from', type=str, default='configs/resunet.py')
    g_load.add_argument('--resume', type=str, default=None)
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


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
Save images on validation.
'''
class image_saver(object):
    def __init__(self, save_path, epoch_length):
        self.save_path = save_path
        self.epoch_length = epoch_length
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, engine, state):

        inputs, target, prediction = state.output[1]

        # Current batch size.
        this_batch_size = len(target)

        # Current batch_num, epoch.
        self._current_batch_num += 1
        if self._current_batch_num==self.epoch_length:
            self._current_epoch += 1
            self._current_batch_num = 0

        # Make directory.
        save_dir = os.path.join(self.save_path, str(self._current_epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Variables to numpy.
        inputs = inputs.cpu().numpy()
        target = target.cpu().numpy()
        prediction = prediction.detach().cpu().numpy()

        # Visualize.
        all_imgs = []
        for i in range(this_batch_size):

            # inputs
            im_i = []
            for x in inputs[i]:
                im_i.append(self._process_slice((x+2)/4.))

            # target
            im_t = [self._process_slice(target[i]/4.)]

            # prediction
            p = prediction[i]
            p[0] = 0
            p[1] *= 1
            p[2] *= 2
            p[3] *= 4
            p = p.max(axis=0)
            im_p = [self._process_slice(p/4.)]

            out_image = np.concatenate(im_i+im_t+im_p, axis=1)
            all_imgs.append(out_image)
        imsave(os.path.join(save_dir,
                            "{}.png".format(self._current_batch_num)),
                            np.vstack(all_imgs))

    def _process_slice(self, s):
        s = np.squeeze(s)
        s = np.clip(s, 0, 1)
        s[0,0]=1
        s[0,1]=0
        return s

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

    '''
    Prepare model. The `resume` arg is able to restore the model,
    its state, and the optimizer's state, whereas `model_from`
    (which is mutually exclusive with `resume`) only loads the
    desired architecture.
    '''
    torch.backends.cudnn.benchmark = False   # profiler
    exp_id = None
    if args.resume is not None:
        saved_dict = torch.load(args.resume)
        exp_id = saved_dict['exp_id']
        # Extract the module string, then turn it into a module.
        # From this, we can invoke the model creation function and
        # load its saved weights.
        module_as_str = saved_dict['module_as_str']
        module = imp.new_module('model_from')
        exec(module_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        if not args.cpu:
            model.cuda(args.gpu_id)
        # Load weights and optimizer state.
        model.load_state_dict(saved_dict['weights'])
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
        optimizer.load_state_dict(saved_dict['optim'])
    else:
        # If `model_from` is a .py file, then we import that as a module
        # and load the model. Otherwise, we assume it's a pickle, and
        # we load it, extract the module contained inside, and load it
        if args.model_from.endswith(".py"):
            module = imp.load_source('model_from', args.model_from)
            module_as_str = open(args.model_from).read()
        else:
            module_as_str = torch.load(args.model_from)['module_as_str']
            module = imp.new_module('model_from')
            exec(module_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        if not args.cpu:
            model.cuda(args.gpu_id)
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    print("Number of parameters: {}".format(count_params(model)))

    '''
    Set up experiment directory.
    '''
    exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
    if exp_id is None:
        exp_id = "{}_{}".format(args.name, exp_time)
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
        if not args.cpu:
            dice = dice.cuda(args.gpu_id)
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
        if not args.cpu:
            b0 = b0.cuda(args.gpu_id)
            b1 = b1.cuda(args.gpu_id)
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
            loss /= len(loss_functions) # average
        return ( loss.data.item(),
                 (batch[0], batch[1], output),
                 metrics(output, batch[1]) )
    evaluator = Evaluator(validation_function)
    
    '''
    Set up logging to screen.
    '''
    bs = args.batch_size
    progress_train = progress_report(prefix=None,
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                            lambda engine, state: evaluator.run(loader_valid))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_valid)
    cpt_handler = ModelCheckpoint(\
                                dirname=path,
                                filename_prefix='weights',
                                n_saved=5,
                                score_function=scoring_function("val_metrics"),
                                atomic=False,
                                exist_ok=True,
                                create_dir=True)
    evaluator.add_event_handler(Events.COMPLETED,
                                cpt_handler,
                                {'model':
                                 {'exp_id': exp_id,
                                  'weights': model.state_dict(),
                                  'module_as_str': module_as_str,
                                  'optim': optimizer.state_dict()}
                                })

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
