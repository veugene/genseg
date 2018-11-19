from datetime import datetime
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

    In args, expecting `resume_from` or `model_from`, `name`, and `save_path`.
    """
    def __init__(self, name, parser):
        self.name = name
        self._epoch = [0]
        
        # Set up model, and optimizers.
        args = parser.parse_args()
        if args.resume_from is None:
            model, optimizer = self._init_state(
                                     model_from=args.model_from,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     opt_kwargs=args.opt_kwargs,
                                     weight_decay=args.weight_decay)
            experiment_path, experiment_id = self._setup_experiment_directory(
                name="{}__{}".format(name, args.name),
                save_path=args.save_path)
            self.experiment_path = experiment_path
            self.experiment_id = experiment_id
            if args.weights_from is not None:
                # Load weights from specified checkpoint.
                self._load_state(load_from=args.weights_from, model=model)
                self._epoch[0] = 0
        else:
            args = self._load_and_merge_args(parser)
            state_file = natsorted([fn for fn in os.listdir(args.resume_from)
                                    if fn.startswith('state_dict_')
                                    and fn.endswith('.pth')])[-1]
            state_from = os.path.join(args.resume_from, state_file)
            print("Resuming from {}.".format(state_from))
            model, optimizer = self._init_state(
                                     model_from=state_from,
                                     optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate,
                                     opt_kwargs=args.opt_kwargs,
                                     weight_decay=args.weight_decay)
            self._load_state(load_from=state_from,
                             model=model, optimizer=optimizer)
            self.experiment_path = args.resume_from
        self.args = args
        self.model = model
        self.optimizer = optimizer
        print("Number of parameters\n"+
              "\n".join([" {} : {}".format(key, count_params(self.model[key]))
                         for key in self.model.keys()]))
    
    def get_model_save_dict(self):
        # For checkpoints.
        model_save_dict = {'dict': {'epoch'        : self._epoch,
                                    'experiment_id': self.experiment_id,
                                    'model_as_str' : self.model_as_str}}
        for key in self.model.keys():
            model_save_dict['dict'][key] = {
                'model_state'    : self.model[key].state_dict(),
                'optimizer_state': self.optimizer[key].state_dict()}
        return model_save_dict
    
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
        # Checkpoint for best model performance.
        checkpoint_best_handler = None
        if evaluator is not None:
            checkpoint_best_handler = ModelCheckpoint(
                                        dirname=self.experiment_path,
                                        filename_prefix='best_state',
                                        n_saved=1,
                                        score_function=score_function,
                                        atomic=True,
                                        create_dir=True,
                                        require_empty=False)
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
        checkpoint_last_handler._iteration = self._epoch[0]
        
        # Function to update state dicts, call checkpoint handlers, and
        # increment epoch count.
        def checkpoint_handler(engine):
            model_save_dict = self.get_model_save_dict()
            checkpoint_last_handler(engine, model_save_dict)
            checkpoint_best_handler(engine, model_save_dict)
            self._increment_epoch(engine)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
        
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
        self.experiment_id = saved_dict['experiment_id']
        self._epoch[0] = saved_dict['epoch'][0]+1
        self.model_as_str = saved_dict['model_as_str']
    
    def load_last_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        model, optimizer = self._load_state(load_from=state_from,
                                            model=self.model,
                                            optimizer=self.optimizer)
        self.model = model
        self.optimizer = optimizer
    
    def load_best_state(self):
        state_file = natsorted([fn for fn in os.listdir(self.experiment_path)
                                if fn.startswith('best_state_dict_')
                                and fn.endswith('.pth')])[-1]
        state_from = os.path.join(self.experiment_path, state_file)
        model, optimizer = self._load_state(load_from=state_from,
                                            model=self.model,
                                            optimizer=self.optimizer)
        self.model = model
        self.optimizer = optimizer

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
        with open(os.path.join(path, "config.py"), 'w') as f:
            f.write(self.model_as_str)
        return path, experiment_id


def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num 