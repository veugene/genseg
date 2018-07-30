import os
import sys
from datetime import datetime
import imp
import numpy as np
import scipy.misc
import torch
from ignite.engine import (Events,
                           Engine)
from ignite.handlers import ModelCheckpoint
from .pytorch import count_params
from .ignite import progress_report


"""
Parse and set up arguments, set up the model and optimizer, and prepare the
folder into which to save the experiment.

In args, expecting `resume_from`, `model_from`, `name`, and `save_path`.
"""
class experiment(object):
    def __init__(self, name, parser):
        self.name = name
        
        # Set up args, model, and optimizer.
        args = parser.parse_args()
        initial_epoch = 0
        if args.resume_from is None:
            model, optimizer = self._init_state(
                                            model_from=args.model_from,
                                            optimizer_name=args.optimizer,
                                            learning_rate=args.learning_rate,
                                            weight_decay=args.weight_decay)
        else:
            args, initial_epoch = self._load_and_merge_args(parser)
            model, optimizer = self._load_state(load_from=args.model_from,
                                                optimizer_name=args.optimizer)
        if not args.cpu:
            model.cuda()
        self.args = args
        self.initial_epoch = initial_epoch
        self.model = model
        self.optimizer = optimizer
        print("Number of parameters: {}".format(count_params(model)))
        
        # Set up experiment directory.
        experiment_path, experiment_id = self._setup_experiment_directory(
            name="{}__{}".format(name, args.name),
            save_path=args.save_path)
        self.experiment_path = experiment_path
        self.experiment_id = experiment_id
        
        # For checkpoints.
        self.model_dict = {
            'experiment_id'   : experiment_id,
            'model_as_str'    : model._model_as_str,
            'model_state'     : model.state_dict(),
            'optimizer_state' : optimizer.state_dict()}
        
    
    def setup_engine(self, function,
                     append=False, prefix=None, epoch_length=None):
        engine = Engine(function)
        fn = "log.txt" if prefix is None else "{}_log.txt".format(prefix)
        progress = progress_report(
            prefix=prefix,
            append=append,
            epoch_length=epoch_length,
            log_path=os.path.join(self.experiment_path, fn))
        engine.add_event_handler(Events.ITERATION_COMPLETED, progress)
        return engine
    
    def setup_checkpoints(self, trainer, evaluator,
                          score_function=None, n_saved=2):
        checkpoint_last_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='state',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  checkpoint_last_handler,
                                  self.model_dict)
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
            optimizer = torch.optim.Adam(betas=(0.9,0.999), **kwargs)
        elif name=='amsgrad':
            optimizer = torch.optim.Adam(betas=(0.9,0.999), amsgrad=True,
                                         **kwargs)
        elif name=='rmsprop':
            optimizer = torch.optim.RMSprop(alpha=0.9, **kwargs)
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
        
        with open(os.path.join(args.resume_from, "cmd.sh"), 'r') as arg_file:
            # Remove first word when parsing arguments from file.
            arg_line = arg_file.read().split(' ')[1:]
            args_from_file = parser.parse_args(arg_line)
            setattr(args_from_file, 'resume_from',
                    getattr(args, 'resume_from'))
            args = args_from_file
            
        # Override loaded arguments with any provided anew.
        args = parser.parse_args(namespace=args)
        
        # TODO: Get last epoch.
            
        return args, initial_epoch

    def _setup_experiment_directory(self, name, save_path):
        experiment_time = "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
        experiment_id   = "{}_{}".format(name, experiment_time)
        path = os.path.join(save_path, experiment_id)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "cmd.sh"), 'w') as f:
            f.write(' '.join(sys.argv))
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

