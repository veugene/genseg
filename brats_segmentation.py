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
from ignite.engines import (Events,
                            Trainer,
                            Evaluator)
from ignite.handlers import ModelCheckpoint

from utils.ignite import (progress_report,
                          metrics_handler,
                          scoring_function)
from utils.metrics import (dice_loss,
                           accuracy)
from utils.data import (data_flow_sampler,
                        preprocessor_brats,
                        masked_view)
from util import count_params
from model import configs
from fcn_maker.model import assemble_resunet
from fcn_maker.blocks import (tiny_block,
                              basic_block)

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation on BRATS 2017.")
    parser.add_argument('--name', type=str, default="brats_seg")
    parser.add_argument('--dataset', type=str, choices=['brats17', 'brats13s'],
                        default='brats17')
    parser.add_argument('--data_dir', type=str, default='/home/eugene/data/')
    parser.add_argument('--save_path', type=str, default='./experiments')
    g_load = parser.add_mutually_exclusive_group(required=False)
    g_load.add_argument('--model_from', type=str, default='configs/resunet.py')
    g_load.add_argument('--resume', type=str, default=None)
    parser.add_argument('--classes', type=str, default='1,2,4',
                        help='Comma-separated list of class labels')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--masked_fraction', type=float, default=0)
    parser.add_argument('--orientation', type=int, default=None)
    parser.add_argument('--batch_size_train', type=int, default=80)
    parser.add_argument('--batch_size_valid', type=int, default=400)
    parser.add_argument('--validate_every', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='RMSprop')
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--nb_io_workers', type=int, default=1)
    parser.add_argument('--nb_proc_workers', type=int, default=2)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--rseed', type=int, default=42)
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
    def __init__(self, save_path, epoch_length, score_function=None):
        self.save_path = save_path
        self.epoch_length = epoch_length
        self.score_function = score_function
        self._max_score = -np.inf
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, engine):
        # If tracking a score, only save whenever a max score is reached.
        if self.score_function is not None:
            score = float(self.score_function(engine))
            if score > self._max_score:
                self._max_score = score
            else:
                return

        # Unpack inputs, outputs.
        inputs, target = engine.state.output[1]
        prediction = engine.state.output[2]

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
                            "{}.jpg".format(self._current_batch_num)),
                            np.vstack(all_imgs))

    def _process_slice(self, s):
        s = np.squeeze(s)
        s = np.clip(s, 0, 1)
        s[0,0]=1
        s[0,1]=0
        return s
    

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'brats17':
        from utils.data import prepare_data_brats17 as prepare_data_brats
    elif args.dataset == 'brats13s':
        from utils.data import prepare_data_brats13s as prepare_data_brats
    else:
        raise ValueError("`dataset` must only be 'brats17' or 'brats13s'")

    orientation = None
    if type(args.orientation) == int:
        orientation = [args.orientation]
            
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    data = prepare_data_brats(path_hgg=os.path.join(args.data_dir, "hgg.h5"),
                              path_lgg=os.path.join(args.data_dir, "lgg.h5"),
                              masked_fraction=args.masked_fraction,
                              orientations=orientation,
                              drop_masked=True,
                              rng=np.random.RandomState(args.rseed))
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
    preprocessor_train = preprocessor_brats(data_augmentation_kwargs=da_kwargs,
                                            h_idx=None, s_idx=0, m_idx=1)
    loader_train = data_flow_sampler(data_train,
                                     sample_random=True,
                                     batch_size=args.batch_size_train,
                                     preprocessor=preprocessor_train,
                                     nb_io_workers=args.nb_io_workers,
                                     nb_proc_workers=args.nb_proc_workers,
                                     rng=np.random.RandomState(args.rseed))
    preprocessor_valid = preprocessor_brats(data_augmentation_kwargs=None,
                                            h_idx=None, s_idx=0, m_idx=1)
    loader_valid = data_flow_sampler(data_valid,
                                     sample_random=True,
                                     batch_size=args.batch_size_valid,
                                     preprocessor=preprocessor_valid,
                                     nb_io_workers=args.nb_io_workers,
                                     nb_proc_workers=0,
                                     rng=np.random.RandomState(args.rseed))

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
            model = getattr(module, 'build_model')()
        else:
            saved_dict = torch.load(args.model_from)
            module_as_str = saved_dict['module_as_str']
            module = imp.new_module('model_from')
            exec(module_as_str, module.__dict__)
            model = getattr(module, 'build_model')()
            model.load_state_dict(saved_dict['weights'])
        if not args.cpu:
            model.cuda(args.gpu_id)
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    print("Number of parameters: {}".format(count_params(model)))

    '''
    Set up experiment directory.
    '''
    exp_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
    if exp_id is None:
        exp_id = args.name
        if not args.no_timestamp:
            exp_id += "_{}".format(args.name, exp_time)
    path = os.path.join(args.save_path, exp_id)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "cmd.sh"), 'w') as f:
        f.write(' '.join(sys.argv))
        
    '''
    Set up loss functions and metrics. Since this is a multiclass problem,
    set up a metrics handler for each output map.
    '''
    labels_str = args.classes.split(",")
    labels = [0] + [int(x) for x in labels_str]
    loss_functions = []
    metrics = {'train': None, 'valid': None}
    for key in metrics.keys():
        metrics_dict = OrderedDict()
        
        # Dice score for every class.
        for idx,l in enumerate(labels):
            dice = dice_loss(l,idx)
            g_dice = dice_loss(target_class=l, target_index=idx,
                               accumulate=True)
            if not args.cpu:
                dice = dice.cuda(args.gpu_id)
                g_dice = g_dice.cuda(args.gpu_id)
            loss_functions.append(dice)
            metrics_dict['dice{}'.format(l)] = g_dice
            
        # Overall tumour Dice.
        g_dice = dice_loss(target_class=labels[1:],
                           target_index=list(range(labels[1],len(labels))),
                           accumulate=True)
        if not args.cpu:
            g_dice = g_dice.cuda(args.gpu_id)
        metrics_dict['dice_tot'] = g_dice
        
        metrics[key] = metrics_handler(metrics_dict)
    
    '''
    Visualize validation outputs.
    '''
    epoch_length = lambda ds, bs : len(ds)//bs + int(len(ds)%bs>0)
    num_batches_valid = epoch_length(data['valid']['s'], args.batch_size_valid)
    image_saver_valid = image_saver(save_path=os.path.join(path, "validation"),
                                    epoch_length=num_batches_valid,
                                score_function=scoring_function("val_metrics"))
                                    
    
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

    def training_function(engine, batch):
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
        with torch.no_grad():
            metrics_dict = metrics['train'](output.detach(), batch[1])
        return loss.item(), batch, output.detach(), metrics_dict
    trainer = Trainer(training_function)

    def validation_function(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            output = model(batch[0])
            loss = 0.
            for i in range(len(loss_functions)):
                loss += loss_functions[i](output, batch[1])
            loss /= len(loss_functions) # average
            metrics_dict = metrics['valid'](output, batch[1])
        return loss.item(), batch, output.detach(), metrics_dict
    evaluator = Evaluator(validation_function)
    
    '''
    Reset global Dice score counts every epoch (or validation run).
    '''
    for l in [str(x) for x in labels[1:]] + ['_tot']:
        func = lambda key : \
            metrics[key].measure_functions['dice{}'.format(l)].reset_counts
        trainer.add_event_handler(Events.EPOCH_STARTED, func('train'))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, func('valid'))
    
    '''
    Set up logging to screen.
    '''
    progress_train = progress_report(prefix=None,
                                     append=args.resume is not None,
                                     log_path=os.path.join(path,
                                                           "log_train.txt"))
    progress_valid = progress_report(prefix="val",
                                     append=args.resume is not None,
                                     log_path=os.path.join(path,
                                                           "log_valid.txt"))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    def evaluator_handler(engine):
        if engine.state.epoch % args.validate_every == 0:
            evaluator.run(loader_valid)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluator_handler)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, image_saver_valid)
    
    '''
    Save a checkpoint every epoch and when encountering the lowest loss.
    '''
    checkpoint_kwargs = {'model': {'exp_id': exp_id,
                                   'weights': model.state_dict(),
                                   'module_as_str': module_as_str,
                                   'optim': optimizer.state_dict()}}
    checkpoint_best_handler = ModelCheckpoint(\
                                dirname=path,
                                filename_prefix='best_weights',
                                n_saved=2,
                                score_function=scoring_function("val_metrics"),
                                atomic=True,
                                exist_ok=True,
                                create_dir=True,
                                require_empty=False)
    evaluator.add_event_handler(Events.COMPLETED,
                                checkpoint_best_handler,
                                checkpoint_kwargs)
    checkpoint_last_handler = ModelCheckpoint(\
                                dirname=path,
                                filename_prefix='weights',
                                n_saved=1,
                                save_interval=1,
                                atomic=True,
                                exist_ok=True,
                                create_dir=True,
                                require_empty=False)
    evaluator.add_event_handler(Events.COMPLETED,
                                checkpoint_last_handler,
                                checkpoint_kwargs)

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
