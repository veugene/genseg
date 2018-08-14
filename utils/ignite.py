from __future__ import (print_function,
                        division)
import os
from collections import OrderedDict
import numpy as np
from scipy.misc import imsave
from tqdm import tqdm
import torch
from ignite.engine import Events


class progress_report(object):
    def __init__(self, epoch_length=None, prefix=None, progress_bar=True,
                 log_path=None, append=False):
        self.epoch_length = epoch_length
        self.prefix = prefix
        self.progress_bar = progress_bar
        self.log_path = log_path
        self._pbar = None
        self.metrics = None
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

