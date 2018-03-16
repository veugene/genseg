from __future__ import print_function
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

'''
Save images on validation.
'''
class image_saver(object):
    def __init__(self, save_path, epoch_length):
        self.save_path = save_path
        self.epoch_length = epoch_length
        self._current_batch_num = 0
        self._current_epoch = 0

    def __call__(self, inputs, target, prediction):
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
            imsave(os.path.join(save_dir,
                                "{}_{}.png"
                                "".format(self._current_batch_num, i)),
                   out_image)

    def _process_slice(self, s):
        s = np.squeeze(s)
        s = np.clip(s, 0, 1)
        s[0,0]=1
        s[0,1]=0
        return s
