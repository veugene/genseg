from collections import (defaultdict,
                         OrderedDict)
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorboardX as tb
import torch
from tqdm import tqdm
from ignite.engine import Events
from ignite.metrics import Metric


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
    """
    metric_name : metric to monitor. This is one of the keys in
        `engine.state.metrics` of the trainer.
    period : only consider saving the best model every this many epochs.
    """
    def __init__(self, metric_name=None, period=1):
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
    
    def _iteration_completed(self, engine, prefix, idx, metric_keys=None):
        output = self._output_transform[idx](engine.state.output)
        if hasattr(engine.state, 'metrics'):
            metrics = OrderedDict([(key, engine.state.metrics[key])
                                    for key in metric_keys])
            for key in metrics.keys():
                metrics[key] = (metrics[key], 1) # Assume count is 1 for each.
            output.update(metrics)
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
    
    def attach(self, engine, prefix=None, output_transform=lambda x:x,
               metric_keys=None):
        '''
        prefix : A string to prefix onto the value names.
        output_transform : function mapping Engine output to a dictionary
            of the form: `(key, (tensor, n_items))`. Stats accumulated for
            each tensor, assuming that it corresponds to `n_items` elements.
        metric_keys : A list of keys (string) for metrics in `engine.metrics`
            to log.
        '''
        idx = len(self._output_transform)
        self._metric_value_dict.append(OrderedDict())
        self._item_counter.append(defaultdict(int))
        self._output_transform.append(output_transform)
        self._epoch.append(self.initial_epoch)
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self._iteration_completed, prefix, idx,
                                 metric_keys)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 self._epoch_completed, idx)
    
    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None


class image_logger(object):
    """
    Expects `output` as a dictionary or list of image lists.
    """
    def __init__(self, num_vis=None, output_transform=lambda x:x,
                 initial_epoch=0, directory=None, summary_tracker=None,
                 suffix=None, min_val=None, max_val=None,
                 output_name='outputs',
                 fontname="LiberationSans-Regular.ttf", fontsize=24):
        self.num_vis = num_vis
        self.directory = directory
        self.summary_tracker = summary_tracker
        self.suffix = suffix
        self.min_val = min_val
        self.max_val = max_val
        self.output_name = output_name
        self.fontname = fontname
        self.fontsize = fontsize
        self._output_transform = output_transform
        self._epoch = initial_epoch
    
    def reset(self):
        self._labels = None
        self._images = []
    
    def collect(self, engine):
        output = self._output_transform(engine.state.output)
        if len(output)==0:
            return
        if isinstance(output, dict):
            labels, images = zip(*output.items())
        else:
            labels, images = None, output
        self._labels = labels
        if len(self._images) < len(images):
            self._images = [[] for _ in images]
        for i, stack in enumerate(images):
            self._images[i].extend(stack)    
    
    def process(self):
        # Select a subset of images.
        if self.num_vis is not None:
            images = [stack[:self.num_vis] for stack in self._images]
        else:
            images = self._images
        
        # Digitize all images.
        images_digitized = []
        for image_stack in images:
            image_stack_digitized = np.zeros_like(image_stack, dtype=np.uint8)
            for i, im in enumerate(image_stack):
                a = im.min() if self.min_val is None else self.min_val
                b = im.max() if self.max_val is None else self.max_val
                step = (b-a)/255. or 1./255
                bins = np.arange(a, a+255*step, step)
                im_d = np.digitize(im, bins).astype(np.uint8)
                image_stack_digitized[i] = im_d
            images_digitized.append(image_stack_digitized)
        
        # Make rows.
        image_rows = []
        for i, row in enumerate(images_digitized):
            arr = np.concatenate(row, axis=1)
            if self._labels is not None:
                arr_pil = Image.fromarray(arr, mode='L')
                draw = ImageDraw.Draw(arr_pil)
                draw.text((0, 0), self._labels[i], fill=255,
                          font=ImageFont.truetype(self.fontname,
                                                  self.fontsize))
                arr = np.array(arr_pil)
            image_rows.append(arr)
        
        # Join rows.
        final_image = np.concatenate(image_rows, axis=0)
        
        # Log to tensorboard.
        if self.summary_tracker is not None:
            self.summary_tracker.summary_writer.add_image(
                self.output_name,
                final_image,
                global_step=self._epoch)
            self.summary_tracker.summary_writer.file_writer.flush()
        
        # Log to file.
        if self.directory is not None:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            _suffix = "_{}".format(self.suffix) if self.suffix else ""
            fn = str(self._epoch)+_suffix+".jpg"
            final_image_pil = Image.fromarray(final_image, mode='L')
            final_image_pil.save(os.path.join(self.directory, fn))
            
        # Update epoch count.
        self._epoch += 1
        
    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, lambda _: self.reset())
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.collect)
        engine.add_event_handler(Events.EPOCH_COMPLETED,
                                 lambda _: self.process())
