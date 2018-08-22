from __future__ import (print_function,
                        division)

import matplotlib
matplotlib.use('agg')   # should be at start; allows use without $DISPLAY
import matplotlib.pyplot as plt

from collections import (defaultdict,
                         OrderedDict)
from datetime import datetime
import imp
import io
import os
import sys

from natsort import natsorted
import numpy as np
import scipy.misc
from scipy.misc import imsave
import tensorflow as tf
import torch
from tqdm import tqdm
from ignite.engine import (Engine,
                           Events)
from ignite.handlers import ModelCheckpoint


"""
Parse and set up arguments, set up the model and optimizer, and prepare the
folder into which to save the experiment.

In args, expecting `resume_from` or `model_from`, `name`, and `save_path`.
"""
class experiment(object):
    def __init__(self, name, parser):
        self.name = name
        
        # Set up args, model, and optimizer.
        args = parser.parse_args()
        self._epoch = [0]
        if args.resume_from is None:
            experiment_path, experiment_id = self._setup_experiment_directory(
                name="{}__{}".format(name, args.name),
                save_path=args.save_path)
            self.experiment_path = experiment_path
            self.experiment_id = experiment_id
            model, optimizer = self._init_state(
                                            model_from=args.model_from,
                                            optimizer_name=args.optimizer,
                                            learning_rate=args.learning_rate,
                                            weight_decay=args.weight_decay)
        else:
            self.experiment_path = args.resume_from
            args = self._load_and_merge_args(parser)
            state_file = natsorted([fn for fn in os.listdir(args.resume_from)
                                    if fn.startswith('state_dict_')
                                    and fn.endswith('.pth')])[-1]
            state_from = os.path.join(args.resume_from, state_file)
            print("Resuming from {}.".format(state_from))
            model, optimizer = self._load_state(load_from=state_from,
                                                optimizer_name=args.optimizer)
        if not args.cpu:
            model.cuda()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        print("Number of parameters: {}".format(count_params(model)))
        
        # For checkpoints.
        self.model_dict = {'dict': {
            'epoch'           : self._epoch,
            'experiment_id'   : self.experiment_id,
            'model_as_str'    : model._model_as_str,
            'model_state'     : model.state_dict(),
            'optimizer_state' : optimizer.state_dict()}}    
    
    def setup_engine(self, function,
                     append=False, prefix=None, epoch_length=None):
        engine = Engine(function)
        fn = "log.txt" if prefix is None else "{}_log.txt".format(prefix)
        progress = progress_report(
            prefix=prefix,
            append=append,
            epoch_length=epoch_length,
            log_path=os.path.join(self.experiment_path, fn))
        progress.attach(engine)
        return engine
    
    def setup_checkpoints(self, trainer, evaluator,
                          score_function=None, n_saved=2):
        # Checkpoint for best model performance.
        checkpoint_best_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='best_state',
                                    n_saved=n_saved,
                                    score_function=score_function,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    checkpoint_best_handler,
                                    self.model_dict)
        
        # Checkpoint at every epoch and increment epoch in `self.model_dict`.
        checkpoint_last_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='state',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        def _on_epoch_completed(engine, model_dict):
            checkpoint_last_handler(engine, model_dict)
            self._increment_epoch(engine)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  _on_epoch_completed,
                                  self.model_dict)
        
        # Setup initial epoch in the training engine.
        trainer.add_event_handler(Events.STARTED,
                                  lambda engine: setattr(engine.state,
                                                         "epoch",
                                                         self._epoch[0]))
                                          
        # Setup initial epoch in the checkpoint handlers.
        checkpoint_last_handler._iteration = self._epoch[0]
        checkpoint_best_handler._iteration = self._epoch[0]
        
        
    def _increment_epoch(self, engine):
        self._epoch[0] += 1
        engine.epoch = self._epoch
        
    def _load_state(self, load_from, optimizer_name):
        '''
        Restore the model, its state, and the optimizer's state.
        '''
        saved_dict = torch.load(load_from)
        
        # Load model.
        module = imp.new_module('module')
        exec(saved_dict['model_as_str'], module.__dict__)
        model = getattr(module, 'build_model')()
        model.load_state_dict(saved_dict['model_state'])
        model._model_as_str = saved_dict['model_as_str']

        # Setup optimizer and load state.
        optimizer = self._get_optimizer(
                                     name=optimizer_name,
                                     params=model.parameters(),
                                     state_dict=saved_dict['optimizer_state'])
        
        # Experiment metadata.
        self.experiment_id = saved_dict['experiment_id']
        self._epoch = saved_dict['epoch']
        
        return model, optimizer
    
    def _init_state(self, model_from, optimizer_name,
                    learning_rate=0., weight_decay=0.):
        '''
        Initialize the model, its state, and the optimizer's state.

        If `model_from` is a .py file, then it is expected to contain a 
        `build_model()` function which is then used to prepare the model.
        If it is not a .py file, it is assumed to be a checkpoint; the model
        saved in the checkpoint is then constructed anew.
        '''
        
        # Build the model.
        if model_from.endswith(".py"):
            model_as_str = open(model_from).read()
        else:
            saved_dict = torch.load(model_from)
            model_as_str = saved_dict['model_as_str']
        module = imp.new_module('module')
        exec(model_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        model._model_as_str = model_as_str
        
        # Setup the optimizer.
        optimizer = self._get_optimizer(name=optimizer_name,
                                        params=model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
        
        return model, optimizer

    def _get_optimizer(self, name, params, lr=0., weight_decay=0.,
                      state_dict=None):
        kwargs = {'params'       : params,
                  'lr'           : lr,
                  'weight_decay' : weight_decay}
        optimizer = None
        if name=='adam':
            optimizer = torch.optim.Adam(betas=(0.5,0.999), **kwargs)
        elif name=='amsgrad':
            optimizer = torch.optim.Adam(betas=(0.5,0.999), amsgrad=True,
                                         **kwargs)
        elif name=='rmsprop':
            optimizer = torch.optim.RMSprop(alpha=0.5, **kwargs)
        elif name=='sgd':
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise NotImplemented("Optimizer {} not supported."
                                "".format(args.optimizer))
        if state_dict is not None:
            optimizer.load_state_dict(state_dict)
        return optimizer
    
    def _load_and_merge_args(self, parser):
        '''
        Loads args from the saved experiment data. Overrides saved args with
        any set anew.
        '''
        args = parser.parse_args()
        initial_epoch = 0
        if not os.path.exists(args.resume_from):
            raise ValueError("Specified resume path does not exist: {}"
                             "".format(args.resume_from))
        
        with open(os.path.join(args.resume_from, "args.txt"), 'r') as arg_file:
            # Remove first word when parsing arguments from file.
            _args = arg_file.read().split('\n')[1:]
            args_from_file = parser.parse_args(_args)
            setattr(args_from_file, 'resume_from',
                    getattr(args, 'resume_from'))
            args = args_from_file
            
        # Override loaded arguments with any provided anew.
        args = parser.parse_args(namespace=args)
        
        return args

    def _setup_experiment_directory(self, name, save_path):
        experiment_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
        experiment_id   = "{}_{}".format(name, experiment_time)
        path = os.path.join(save_path, experiment_id)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.txt"), 'w') as f:
            f.write('\n'.join(sys.argv))
        return path, experiment_id


"""
Save sets of 2D images to file.

save_path : path to save the images to (a new folder is created on every call).
name_images : the name of the attribute, in the Engine state, which contains
    the tuple of image stacks to save.
save_every : save images every `save_every` number of calls.
score_function : a `scoring_function` that returns a score, given the Engine
    state; save images only whenever a new max score is attained.
"""
# TODO: implement per epoch evaluation and saving.
class image_saver(object):
    def __init__(self, save_path, name_images, save_every=1,
                 score_function=None):
        self.save_path      = save_path
        self.name_images    = name_images
        self.save_every     = save_every
        self.score_function = score_function
        self._max_score     = -np.inf
        self._call_counter  = 0

    def __call__(self, engine):
        self._call_counter += 1
        if self._call_counter%self.save_every:
            return
        
        # If tracking a score, only save whenever a max score is reached.
        if self.score_function is not None:
            score = float(self.score_function(engine))
            if score > self._max_score:
                self._max_score = score
            else:
                return
        
        # Get images to save.
        images = getattr(engine.state, self.name_images)
        if not isinstance(images, tuple):
            raise ValueError("Images in `engine.state.{}` should be stored "
                "as a tuple of image stacks.".format(self.name_images))
        n_images = len(images[0])
        shape = np.shape(images[0])
        for stack in images:
            if len(stack)!=n_images:
                raise ValueError("Every stack of images in the "
                    "`engine.state.{}` tuple must have the same length."
                    "".format(self.name_images))
            stack_shape = np.shape(stack)
            if stack_shape!=shape:
                raise ValueError("All images must have the same shape.")
            if len(stack_shape)!=3:
                raise ValueError("All image stacks must be 3-dimensional "
                    "(stacks of 2D images).")
        
        
        # Make sub-directory.
        save_dir = os.path.join(self.save_path,
                                "{}".format(self._call_counter))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Concatenate images across sets and save to disk.
        for i, im_set in enumerate(zip(*images)):
            row = np.concatenate(im_set, axis=1)
            im = scipy.misc.toimage(row, high=255., low=0.)
            im.save(os.path.join(save_dir, "{}.jpg".format(i)))


class progress_report(object):
    """
    Log progress to screen and/or file. Relies on `Metric` objects to
    accumulate stats in the way that they should be reported every
    iteration.
    """
    def __init__(self, epoch_length=None, prefix=None, progress_bar=True,
                 log_path=None, append=False):
        self.epoch_length = epoch_length
        self.prefix = prefix
        self.progress_bar = progress_bar
        self.log_path = log_path
        self._pbar = None
        if log_path is not None and not append:
            # If we're not resuming the experiment,
            # make the log file blank.
            open(log_path, 'w').close()

    def log_string(self, desc, metrics, file=None):
        mstr = " ".join(["{}={:.3}".format(*x) for x in metrics.items()])
        desc = "{}: {}".format(desc, mstr) if len(desc) else mstr
        print(desc, file=file)

    def __call__(self, engine):
        # If no epoch length is defined, try to get it from the data loader.
        if self.epoch_length is None:
            if hasattr(engine.state.dataloader, '__len__'):
                self.epoch_length = len(engine.state.dataloader)
            else:
                self.epoch_length = np.inf
        
        # Setup prefix.
        prefix = ""
        if self.prefix is not None:
            prefix = "{}_".format(self.prefix)
        
        # Print to screen.
        desc = ""
        metrics = {}
        if hasattr(engine.state, 'metrics'):
            metrics = dict([(prefix+key, val.item())
                            if isinstance(val, torch.Tensor)
                            else (prefix+key, val)
                            for key, val in engine.state.metrics.items()])
        if hasattr(engine.state, 'epoch'):
            desc += "Epoch {}".format(engine.state.epoch)
        if self.progress_bar:
            if self._pbar is None:
                self._pbar = tqdm(total=self.epoch_length, desc=desc)
            self._pbar.set_postfix(metrics)
            self._pbar.update(1)
            if engine.state.iteration % self.epoch_length == 0:
                self._pbar.close()
                self._pbar = None
                if self.log_path is not None:
                    with open(self.log_path, 'a') as logfile:
                        self.log_string(desc, metrics, file=logfile)
        else:
            self.log_string(desc, metrics)
    
    def attach(self, engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)


class scoring_function(object):
    def __init__(self, metric_name=None, period=1):
        """
        metric_name : metric to monitor. This is one of the keys in
          `engine.state.metrics` of the trainer.
        period : only consider saving the best model every
          this many epochs.
        """
        if metric_name is None:
            metric_name = 'val_loss'
        self.metric_name = metric_name
        self.period = period
        self.epochs_since_last_save = 0
        
    def __call__(self, engine):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            quantity = engine.state.metrics[self.metric_name]
            # Since we're always trying to minimize things, return
            # the -ve of whatever that is.
            return -quantity
        else:
            return -np.inf


def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num 


class summary_tracker(object):
    """
    Accumulate statistics.
    
    path : path to save tensorboard event files to.
    output_transform : function mapping Engine output to
        `value_dict` - Dict of torch tensors for which to accumulate stats.
        `n_items`    - # of elements in the batch used to compute values.
    """
    def __init__(self, path, output_transform=lambda x: x):
        self.path = path
        self.output_transform = output_transform
        self.summary_writer = tf.summary.FileWriter(path)
        self._metric_value_dict = OrderedDict()
        self._item_counter = defaultdict(int)
        self._epoch = 1
    
    def _iteration_completed(self, engine, prefix=None):
        output = self.output_transform(engine.state.output)
        self._update(output, prefix)
    
    def _epoch_completed(self, engine):
        self._write(self._epoch)
        self._epoch += 1
    
    def _update(self, output, prefix=None):
        _value_dict = output
        value_dict = OrderedDict()
        for key in _value_dict:
            # Convert torch tensors to numpy.
            val, count = _value_dict[key]
            if isinstance(val, torch.Tensor):
                value_dict[key] = (val.cpu().numpy(), count)
        for _key, (val, count) in value_dict.items():
            # Prefix key, if requested.
            key = _key
            if prefix is not None:
                key = "{}_{}".format(prefix, key)
                
            # Initialize
            init_keys = None
            if len(val.shape)==3:
                init_keys = [key+'_image_mean', key+'_image_std']
            elif len(val.shape)==0:
                init_keys = [key]
            else:
                init_keys = [key+'_min', key+'_max', key+'_mean', key+'_std']
            for k in init_keys:
                if k not in self._metric_value_dict:
                    self._metric_value_dict[k] = np.zeros(val.shape[1:],
                                                          dtype=np.float32)

            # Matrix: log image. Update mean, var. Online - Welford's method.
            if len(val.shape)==3:
                mean = self._metric_value_dict[key+'_image_mean']
                var  = self._metric_value_dict[key+'_image_std']
                count = 0
                for i, elem in enumerate(val):
                    # Update mean, var.
                    count += 1
                    old_mean = mean
                    mean += (elem-mean)/float(count)
                    var  += (elem-mean)*(elem-old_mean)
                self._metric_value_dict[key+'_image_mean'] = mean
                self._metric_value_dict[key+'_image_std']  = var
            
            # Non-image.
            else:
                a, b = self._item_counter[key], count
                labels, reductions = [], []
                if len(val.shape)==0:       # Scalar.
                    labels.append('')
                    reductions.append(lambda x: x)
                else:
                    axes = tuple(range(1, len(val.shape)))
                    labels.extend(['_min', '_max', '_mean', '_std'])
                    reductions.extend([lambda x: np.amin(x, axis=axes),
                                       lambda x: np.amax(x, axis=axes),
                                       lambda x: np.mean(x, axis=axes),
                                       lambda x:  np.std(x, axis=axes)])
                for l, f in zip(labels, reductions):
                    old_mean = self._metric_value_dict[key+l]
                    reduced_val = np.mean(f(val))
                    mean = (old_mean*a + reduced_val*b) / float(a+b)
                    self._metric_value_dict[key+l] = mean
                
            # Increment counter.
            self._item_counter[key] += count
    
    def _write(self, epoch=None):
        '''
        Write summaries to event. Increments step count.
        
        global_step : typically, the current epoch.
        '''
        for key, val in self._metric_value_dict.items():
            s_value = None
            if key.endswith('_image_std') or key.endswith('_image_mean'):
                if key.endswith('_image_std'):
                    # Turn variance into standard deviation
                    val = np.sqrt(val)
                s = io.BytesIO()
                plt.imsave(s, val, format='png')
                img = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=val.shape[0],
                                       width=val.shape[1])
                s_value = tf.Summary.Value(tag=key, image=img)
            else:
                s_value = tf.Summary.Value(tag=key, simple_value=val)
            self.summary_writer.add_summary(tf.Summary(value=[s_value]),
                                            global_step=epoch)
            self.summary_writer.flush()
        self._item_counter = defaultdict(int)
        self._metric_value_dict = OrderedDict()
    
    def attach(self, engine, prefix=None):
        '''
        prefix : A string to prefix onto the value names.
        '''
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._iteration_completed, prefix)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._epoch_completed)
    
    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.flush()
            self.summary_writer.close()
            self.summary_writer = None
