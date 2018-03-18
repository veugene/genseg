from __future__ import (print_function,
                        division)
import os
import numpy as np
from collections import OrderedDict
from scipy.misc import imsave
from tqdm import tqdm


class progress_report(object):
    def __init__(self, epoch_length=None, prefix=None, progress_bar=True,
                 log_path=None):
        self.epoch_length = epoch_length
        self.prefix = prefix
        self.progress_bar = progress_bar
        self.log_path = log_path
        self._pbar = None
        self.metrics = None

    def log_string(self, desc, metrics, file=None):
        mstr = " ".join(["{}={:.3}".format(*x) for x in metrics.items()])
        desc = "{}: {}".format(desc, mstr) if len(desc) else mstr
        print(desc, file=file)
       
    def __call__(self, engine, state):
        if self.epoch_length is None:
            # If no epoch length is defined, see if we can pull it
            # out of the data loader.
            if hasattr(state.dataloader, '__len__'):
                self.epoch_length = len(state.dataloader)
            else:
                self.epoch_length = np.inf
        prefix = ""
        if self.prefix is not None:
            prefix = "{}_".format(self.prefix)
        if (state.iteration-1) % self.epoch_length == 0:
            # Every time we're at the start of the next epoch, reset
            # the statistics.
            self.metrics = OrderedDict(((prefix+'loss', 0),))
            setattr(state, prefix+"metrics", self.metrics)
        iter_history = state.output # only this iteration's history
        loss = iter_history[0]
        self.metrics[prefix+'loss'] += loss
        for name, val in iter_history[-1].items():
            name = prefix+name
            if name not in self.metrics:
                self.metrics[name] = 0
            self.metrics[name] += val
        # Average metrics over the epoch thus far.
        metrics = OrderedDict()
        denom = ((float(state.iteration)-1) % self.epoch_length) + 1
        for name in self.metrics:
            metrics[name] = self.metrics[name] / denom
        # Print to screen.
        desc = ""
        if hasattr(state, 'epoch'):
            desc += "Epoch {}".format(state.epoch)
        if self.progress_bar:
            if self._pbar is None:
                self._pbar = tqdm(total=self.epoch_length, desc=desc)
            self._pbar.set_postfix(metrics)
            self._pbar.update(1)
            if state.iteration % self.epoch_length == 0:
                self._pbar.close()
                self._pbar = None
                if self.log_path is not None:
                    with open(self.log_path, 'a') as logfile:
                        self.log_string(desc, metrics, file=logfile)
        else:
            self.log_string(desc, metrics)


class metrics_handler(object):
    def __init__(self, measure_functions_dict=None):
        self.measure_functions = measure_functions_dict
        if measure_functions_dict is None:
            self.measure_functions = OrderedDict()

    def __call__(self, output, target):
        measures = OrderedDict()
        for m in self.measure_functions:
            measures[m] = self.measure_functions[m](output, target).data.item()
        return measures

    def add(self, metric_name, function):
        self.measure_functions[metric_name] = function


class scoring_function(object):
    def __init__(self, metrics_name, loss_name=None, period=1):
        """
        metrics_name : the name of the dictionary in the state
          object storing the metrics to monitor.
        loss_name : loss to monitor. This is one of the keys in
          state.output[-1] of the trainer.
        period : only consider saving the best model every
          this many epochs.
        """
        self.metrics_name = metrics_name
        if loss_name is None:
            loss_name = 'val_loss'
        self.loss_name = loss_name
        self.period = period
        self.epochs_since_last_save = 0
        
    def __call__(self, state):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            quantity = getattr(state, self.metrics_name)[self.loss_name]
            # Since we're always trying to minimize things, return
            # the -ve of whatever that is.
            return -quantity
        else:
            return -np.inf

