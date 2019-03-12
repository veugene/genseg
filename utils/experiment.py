import imp
import os
import sys

from natsort import natsorted
import torch
import numpy as np
import json
from ignite.engine import (Engine,
                           Events)
from ignite.handlers import ModelCheckpoint

from .trackers import progress_report


class experiment(object):
    """
    Parse and set up arguments, set up the model and optimizer, and prepare 
    the folder into which to save the experiment.

    In args, expecting `model_from` and `path`.
    """
    def __init__(self, parser):
        self._epoch = [0]
        args = parser.parse_args()
        self.experiment_path = args.path
        
        # Make experiment directory, if necessary.
        if not os.path.exists(args.path):
            os.makedirs(args.path)
            
        # Does the experiment directory already contain state files?
        state_file_list = natsorted([fn for fn in os.listdir(args.path)
                                     if fn.startswith('state_dict_')
                                     and fn.endswith('.pth')])
        
        # If yes, resume; else, initialize a new experiment.
        if os.path.exists(args.path) and len(state_file_list):
            # RESUME old experiment
            args = self._load_and_merge_args(parser)
            state_file = state_file_list[-1]
            state_from = os.path.join(args.path, state_file)
            print("Resuming from {}.".format(state_from))
            model, optimizer = self._init_state(
                                     model_from=state_from,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     opt_kwargs=args.opt_kwargs,
                                     weight_decay=args.weight_decay)
            self._load_state(load_from=state_from,
                             model=model, optimizer=optimizer)
        else:
            # INIT new experiment
            args = self._load_and_merge_args(parser)
            model, optimizer = self._init_state(
                                     model_from=args.model_from,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     opt_kwargs=args.opt_kwargs,
                                     weight_decay=args.weight_decay)
            if args.weights_from is not None:
                # Load weights from specified checkpoint.
                self._load_state(load_from=args.weights_from, model=model)
                self._epoch[0] = 0
            with open(os.path.join(args.path, "args.txt"), 'w') as f:
                f.write('\n'.join(sys.argv))
            with open(os.path.join(args.path, "config.py"), 'w') as f:
                f.write(self.model_as_str)
        
        # Initialization complete. Store objects, etc.
        self.args = args
        self.model = model
        self.optimizer = optimizer
        print("Number of parameters\n"+
              "\n".join([" {} : {}".format(key, count_params(self.model[key]))
                         for key in self.model.keys()]))
    
    def setup_engine(self, function,
                     append=True, prefix=None, epoch_length=None):
        engine = Engine(function)
        fn = "log.txt" if prefix is None else "{}_log.txt".format(prefix)
        progress = progress_report(
            prefix=prefix,
            append=append,
            epoch_length=epoch_length,
            log_path=os.path.join(self.experiment_path, fn))
        progress.attach(engine)
        return engine
    
    def setup_checkpoints(self, trainer, evaluator, score_function, n_saved=2):
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
        
        # Checkpoint for best model performance.
        checkpoint_best_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='best_state',
                                    n_saved=1,
                                    score_function=score_function,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        
        # Save checkpoint histories in a lightweight checkpoint.
        # If experiment state is resumed, so are these histories.
        checkpoint_hist_handler = ModelCheckpoint(
                                    dirname=self.experiment_path,
                                    filename_prefix='checkpoint_history',
                                    n_saved=n_saved,
                                    save_interval=1,
                                    atomic=True,
                                    create_dir=True,
                                    require_empty=False)
        
        # The lists of saved checkpoints is stored in the experiment object.
        # If experiment state is resumed, so are these lists.
        # 
        # Monkey-patch the checkpoint handlers to use _these_ lists rather
        # than their own which are always initialized empty, even when
        # resuming an experiment.
        checkpoint_last_handler._saved = self._checkpoint_saved_last
        checkpoint_best_handler._saved = self._checkpoint_saved_best
        
        # Track epoch in the experiment object, the training engine state,
        # and the checkpoint handlers.
        # 
        # This allows the correct epoch to be reported, training to stop after
        # the correct number of epochs, resumed experiments to start on the
        # correct epoch, and saved states to be enumerated with the correct
        # epoch.
        trainer.add_event_handler(Events.STARTED,
                                  lambda engine: setattr(engine.state,
                                                         "epoch",
                                                         self._epoch[0]))
        checkpoint_last_handler._iteration = self._epoch[0]
        checkpoint_best_handler._iteration = self._epoch[0]
        checkpoint_hist_handler._iteration = self._epoch[0]
        
        # Function to update state dicts, call checkpoint handler, and
        # increment epoch count. Runs at the end of each epoch.
        # 
        # Also stores the '_saved' list of previous checkpoints from all
        # checkpoint handlers, so that previous checkpoints could be cleaned
        # up when an experiment is resumed.
        # 
        # To support saving some problematic layers (eg. layer normalization),
        # `state_dict()` is called every time a state dict is needed (every
        # time that a state is saved).
        # 
        # NOTE: `checkpoint_handler` only uses `engine` by applying
        # `score_function` to it -- so only the `checkpoint_best_handler`
        # needs an enginer (the `evaluator` engine). This engine can safely
        # be passed to the `checkpoint_last_handler`, too, since it will not
        # be used there.
        def call_checkpoint_handlers(engine):
            self._epoch[0] = trainer.state.epoch  # Update the epoch count.
            model_save_dict = {'dict': {
                'epoch'        : self._epoch,
                'model_as_str' : self.model_as_str
                }}
            for key in self.model.keys():
                model_save_dict['dict'][key] = {
                    'model_state'     : self.model[key].state_dict(),
                    'optimizer_state' : self.optimizer[key].state_dict()}
            checkpoint_last_handler(engine, model_save_dict)                
            checkpoint_best_handler(engine, model_save_dict)
            hist_save_dict = {'dict': {
                'last_checkpoint' : checkpoint_last_handler._saved,
                'best_checkpoint' : checkpoint_best_handler._saved
                }}
            checkpoint_hist_handler(engine, hist_save_dict)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                    call_checkpoint_handlers)
        
    
    def get_epoch(self):
        return self._epoch[0]
    
    def _init_state(self, model_from, optimizer_name, learning_rate=0.,
                    opt_kwargs=None, weight_decay=0.):
        '''
        Initialize the model, its state, and the optimizer's state.

        If `model_from` is a .py file, then it is expected to contain a 
        `build_model()` function which is then used to prepare the model.
        If it is not a .py file, it is assumed to be a checkpoint; the model
        saved in the checkpoint is then constructed anew.
        '''
        
        # Build the model.
        if os.path.isdir(model_from):
            ##we have a directory, need to grab .py file
            import glob
            model_from = glob.glob(os.path.join(model_from, "*.py"))[0]

        if model_from.endswith(".py"):
            self.model_as_str = open(model_from).read()
        else:
            saved_dict = torch.load(model_from)
            self.model_as_str = saved_dict['model_as_str']
        module = imp.new_module('module')
        exec(self.model_as_str, module.__dict__)
        model = getattr(module, 'build_model')()
        if not isinstance(model, dict):
            model = {'model': model}
        
        # If optimizer_name is in JSON format, convert string to dict.
        try:
            optimizer_name = json.loads(optimizer_name)
        except ValueError:
            # Not in JSON format; keep as a string.
            optimizer_name = optimizer_name
        
        # Setup the optimizer.
        optimizer = {}
        for key in model.keys():
            def parse(arg):
                # Helper function for args when passed as dict, with
                # model names as keys.
                if isinstance(arg, dict) and key in arg:
                    return arg[key]
                return arg
            model[key].cuda()
            optimizer[key] = self._get_optimizer(
                                            name=parse(optimizer_name),
                                            params=model[key].parameters(),
                                            lr=parse(learning_rate),
                                            opt_kwargs=parse(opt_kwargs),
                                            weight_decay=weight_decay)
        
        # Store the lists of past checkpoints in the experiment object.
        # (checkpoint handler internals -- used for resuming handler state).
        self._checkpoint_saved_last = []
        self._checkpoint_saved_best = []
        
        return model, optimizer
    
    def _load_state(self, load_from, model, optimizer=None):
        '''
        Restore the model, its state, and the optimizer's state.
        '''
        saved_dict = torch.load(load_from)
        for key in model.keys():
            model[key].load_state_dict(saved_dict[key]['model_state'])
            if optimizer is not None:
                optimizer[key].load_state_dict(
                                   saved_dict[key]['optimizer_state'])
        
        # Experiment metadata.
        self._epoch[0] = saved_dict['epoch'][0]
        
        # Store the lists of past checkpoints in the experiment object.
        # (checkpoint handler internals -- used for resuming handler state).
        def clean(checkpoint_list):
            # Keep only the paths that exist.
            valid_checkpoint_list = []
            for priority, paths in checkpoint_list:
                if np.all([os.path.exists(p) for p in paths]):
                    valid_checkpoint_list.append((priority, paths))   
            return valid_checkpoint_list
        hist_file_list = natsorted(
            [fn for fn in os.listdir(self.experiment_path)
             if fn.startswith('checkpoint_history_dict_')
             and fn.endswith('.pth')])
        if len(hist_file_list):
            hist_path = os.path.join(self.experiment_path, hist_file_list[-1])
            hist_dict = torch.load(hist_path)
            self._checkpoint_saved_last = clean(hist_dict['last_checkpoint'])
            self._checkpoint_saved_best = clean(hist_dict['best_checkpoint'])
    
    def load_last_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        self._load_state(load_from=state_from,
                         model=self.model,
                         optimizer=self.optimizer)
    
    def load_best_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('best_state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        self._load_state(load_from=state_from,
                         model=self.model,
                         optimizer=self.optimizer)

    def _get_optimizer(self, name, params, lr=0., opt_kwargs=None, 
                       weight_decay=0.):
        kwargs = {'params'       : [p for p in params if p.requires_grad],
                  'lr'           : lr,
                  'weight_decay' : weight_decay}
        optimizer = None
        if name=='adam' or name=='amsgrad':
            if opt_kwargs is None:
                opt_kwargs = {'betas': (0.5, 0.999)}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.Adam(amsgrad=bool(name=='amsgrad'),
                                         **kwargs)
        elif name=='rmsprop':
            if opt_kwargs is None:
                opt_kwargs = {'alpha': 0.5}
            kwargs.update(opt_kwargs)
            optimizer = torch.optim.RMSprop(**kwargs)
        elif name=='sgd':
            optimizer = torch.optim.SGD(**kwargs)
        else:
            raise ValueError("Optimizer {} not supported.".format(name))
        return optimizer
    
    def _load_and_merge_args(self, parser):
        '''
        Loads args from the saved experiment data. Overrides saved args with
        any set anew.
        '''
        args = parser.parse_args()
        initial_epoch = 0        
        with open(os.path.join(args.path, "args.txt"), 'r') as arg_file:
            # Remove first word when parsing arguments from file.
            _args = arg_file.read().split('\n')[1:]
            args_from_file = parser.parse_args(_args)
            args = args_from_file

        # If loading config file and getting args from there, merge with loaded arguments (newly passed arguments
        # override loaded arguments).
        path_for_args = os.path.join(args.model_from, "args.txt")
        if os.path.exists(path_for_args):
            with open(path_for_args, 'r') as f:
                saved_args = f.read().split('\n')[1:]
                saved_args = [s for s in saved_args if s != '']
                args = parser.parse_args(args=saved_args)

            args = parser.parse_args(namespace=args)

            
        # Override loaded arguments with any provided anew.
        args = parser.parse_args(namespace=args)
        
        return args


def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num 