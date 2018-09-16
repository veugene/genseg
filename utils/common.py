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
from PIL import Image, ImageDraw
import tensorboardX as tb
import torch
from tqdm import tqdm
from ignite.engine import (Engine,
                           Events)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric


"""
Parse and set up arguments, set up the model and optimizer, and prepare the
folder into which to save the experiment.

In args, expecting `resume_from` or `model_from`, `name`, and `save_path`.
"""
class experiment(object):
    def __init__(self, name, parser):
        self.name = name
        
        # Set up args, model, and optimizers.
        args = parser.parse_args()
        self._epoch = [0]
        if args.resume_from is None:
            experiment_path, experiment_id = self._setup_experiment_directory(
                name="{}__{}".format(name, args.name),
                save_path=args.save_path)
            self.experiment_path = experiment_path
            self.experiment_id = experiment_id
            model, optimizer = self._init_state(model_from=args.model_from,
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
        self.args = args
        self.model = model
        self.optimizer = optimizer
        if not args.cpu:
            for model in self.model.values():
                model.cuda()
        print("Number of parameters\n"+
              "\n".join([" {} : {}".format(key, count_params(self.model[key]))
                         for key in self.model.keys()]))
        
        # For checkpoints.
        self.model_save_dict = {'epoch'        : self._epoch,
                                'experiment_id': self.experiment_id,
                                'model_as_str' : self.model_as_str}
        for key in self.model.keys():
            self.model_save_dict[key] = {
                'model_state'    : self.model[key].state_dict(),
                'optimizer_state': self.optimizer[key].state_dict()}
    
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
    
    def setup_checkpoints(self, trainer, evaluator=None,
                          score_function=None, n_saved=2):
        if evaluator is not None:
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
                                        self.model_save_dict)
            checkpoint_best_handler._iteration = self._epoch[0]
        
        # Checkpoint at every epoch and increment epoch in
        # `self.model_save_dict`.
        checkpoint_last_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='state',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        def _on_epoch_completed(engine, model_save_dict):
            checkpoint_last_handler(engine, model_save_dict)
            self._increment_epoch(engine)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  _on_epoch_completed,
                                  self.model_save_dict)
        checkpoint_last_handler._iteration = self._epoch[0]
        
        # Setup initial epoch in the training engine.
        trainer.add_event_handler(Events.STARTED,
                                  lambda engine: setattr(engine.state,
                                                         "epoch",
                                                         self._epoch[0]))

    def get_epoch(self):
        return self._epoch[0]
        
    def _increment_epoch(self, engine):
        self._epoch[0] += 1
        engine.epoch = self._epoch
        
    def _load_state(self, load_from, optimizer_name):
        '''
        Restore the model, its state, and the optimizer's state.
        '''
        saved_dict = torch.load(load_from)
        
        # Build model according to saved code.
        module = imp.new_module('module')
        exec(saved_dict['model_as_str'], module.__dict__)
        model = getattr(module, 'build_model')()
        if not isinstance(model, dict):
            model = {'dict': model}

        # Load model and optimizer state.
        optimizer = {}
        for key in model.keys():
            model[key].load_state_dict(saved_dict[key]['model_state'])
            optimizer[key] = self._get_optimizer(
                               name=optimizer_name,
                               params=model[key].parameters(),
                               state_dict=saved_dict[key]['optimizer_state'])
        
        # Experiment metadata.
        self.experiment_id = saved_dict['experiment_id']
        self._epoch[0] = saved_dict['epoch'][0]+1
        self.model_as_str = model_as_str
        
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
            self.model_as_str = open(model_from).read()
        else:
            saved_dict = torch.load(model_from)
            self.model_as_str = saved_dict['model_as_str']
        module = imp.new_module('module')
        exec(self.model_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        if not isinstance(model, dict):
            model = {'dict': model}
        
        # Setup the optimizer.
        optimizer = {}
        for key in model.keys():
            optimizer[key] = self._get_optimizer(
                                            name=optimizer_name,
                                            params=model[key].parameters(),
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
        experiment_id   = "{}_{}".format(experiment_time, name)
        path = os.path.join(save_path, experiment_id)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.txt"), 'w') as f:
            f.write('\n'.join(sys.argv))
        return path, experiment_id


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
    """
    def __init__(self, path, initial_epoch=0):
        self.path = path
        self.initial_epoch = initial_epoch
        self.summary_writer = tb.SummaryWriter(path)
        self._metric_value_dict = []
        self._item_counter = []
        self._output_transform = []
        self._epoch = []
    
    def _iteration_completed(self, engine, prefix, idx):
        output = self._output_transform[idx](engine.state.output)
        self._update(output, prefix, idx)
    
    def _epoch_completed(self, engine, idx):
        self._write(self._epoch[idx], idx)
        self._epoch[idx] += 1
    
    def _update(self, output, prefix, idx):
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
            elif len(val.shape) in [0, 1]:
                init_keys = [key]
            else:
                init_keys = [key+'_min', key+'_max', key+'_mean', key+'_std']
            for k in init_keys:
                if k not in self._metric_value_dict[idx].keys():
                    self._metric_value_dict[idx][k] = np.zeros(val.shape[1:],
                                                            dtype=np.float32)

            # Matrix: log image. Update mean, var. Online - Welford's method.
            if len(val.shape)==3:
                mean = self._metric_value_dict[idx][key+'_image_mean']
                var  = self._metric_value_dict[idx][key+'_image_std']
                count = 0
                for i, elem in enumerate(val):
                    # Update mean, var.
                    count += 1
                    old_mean = mean
                    mean += (elem-mean)/float(count)
                    var  += (elem-mean)*(elem-old_mean)
                self._metric_value_dict[idx][key+'_image_mean'] = mean
                self._metric_value_dict[idx][key+'_image_std']  = var
            
            # Non-image.
            else:
                a, b = self._item_counter[idx][key], count
                labels, reductions = [], []
                if len(val.shape) in [0, 1]:       # Scalar.
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
                    old_mean = self._metric_value_dict[idx][key+l]
                    reduced_val = np.mean(f(val))
                    mean = (old_mean*a + reduced_val*b) / float(a+b)
                    self._metric_value_dict[idx][key+l] = mean
                
            # Increment counter.
            self._item_counter[idx][key] += count
    
    def _write(self, epoch, idx):
        '''
        Write summaries to event. Increments step count.
        
        global_step : typically, the current epoch.
        '''
        for key, val in self._metric_value_dict[idx].items():
            s_value = None
            if key.endswith('_image_std') or key.endswith('_image_mean'):
                if key.endswith('_image_std'):
                    # Turn variance into standard deviation
                    val = np.sqrt(val)
                self.summary_writer.add_image(tag=key,
                                              img_tensor=val,
                                              global_step=epoch)
            else:
                self.summary_writer.add_scalar(tag=key,
                                               scalar_value=val,
                                               global_step=epoch)
            self.summary_writer.file_writer.flush()
        self._item_counter[idx] = defaultdict(int)
        self._metric_value_dict[idx] = OrderedDict()
    
    def attach(self, engine, prefix=None, output_transform=lambda x:x):
        '''
        prefix : A string to prefix onto the value names.
        output_transform : function mapping Engine output to
        `value_dict` - Dict of torch tensors for which to accumulate stats.
        `n_items`    - # of elements in the batch used to compute values.
        '''
        idx = len(self._output_transform)
        self._metric_value_dict.append(OrderedDict())
        self._item_counter.append(defaultdict(int))
        self._output_transform.append(output_transform)
        self._epoch.append(self.initial_epoch)
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._iteration_completed, prefix, idx)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._epoch_completed, idx)
    
    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None


class image_logger(Metric):
    """
    Expects `output` as a dictionary or list of image lists.
    """
    def __init__(self, num_vis, output_transform=lambda x:x, initial_epoch=0,
                 summary_tracker=None, min_val=None, max_val=None):
        super(image_logger, self).__init__(output_transform)
        self.num_vis = num_vis
        self.min_val = min_val
        self.max_val = max_val
        self.summary_tracker = summary_tracker
        self._epoch = initial_epoch
    
    def reset(self):
        self._labels = None
        self._images = []
    
    def update(self, output):
        if isinstance(output, dict):
            labels, images = zip(*output.items())
        else:
            labels, images = None, output
        self._labels = labels
        if len(self._images) < len(images):
            self._images = [[] for _ in images]
        for i, stack in enumerate(images):
            self._images[i].extend(stack)    
    
    def compute(self):
        # Select a subset of images.
        images = [stack[:self.num_vis] for stack in self._images]
        
        # Digitize all images.
        images_digitized = []
        for image_stack in images:
            image_stack_digitized = np.zeros_like(image_stack, dtype=np.uint8)
            for i, im in enumerate(image_stack):
                a = im.min() if self.min_val is None else self.min_val
                b = im.max() if self.max_val is None else self.max_val
                step = (b-a)/255.
                bins = np.arange(a, a+255*step, step)
                im_d = np.digitize(im, bins).astype(np.uint8)
                image_stack_digitized[i] = im_d
            images_digitized.append(image_stack_digitized)
        
        # Make columns.
        image_columns = []
        for i, column in enumerate(images_digitized):
            col = np.concatenate(column, axis=0)
            if self._labels is not None:
                col_pil = Image.fromarray(col, mode='L')
                draw = ImageDraw.Draw(col_pil)
                draw.text((0, 0), self._labels[i], fill=255)
                col = np.array(col_pil)
            image_columns.append(col)
        
        # Join columns.
        final_image = np.concatenate(image_columns, axis=1)
        
        # Log to tensorboard.
        if self.summary_tracker is not None:
            self.summary_tracker.summary_writer.add_image(
                'outputs',
                final_image,
                global_step=self._epoch)
            self.summary_tracker.summary_writer.file_writer.flush()
            
        # Update epoch count.
        self._epoch += 1
        
        return final_image
    