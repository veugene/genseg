from __future__ import (print_function,
                        division)
import torch
from ignite.engine import Events
from ignite.metrics import Metric
from fcn_maker.loss import dice_loss as _dice_loss


class metric(Metric):
    """
    Just like ignite's `Metric` except that it computes the metric every
    iteration (not only every epoch) and stores the result in the engine's
    state.
    """
    def __init__(self, *args, **kwargs):
        super(metric, self).__init__(*args, **kwargs)
        self._cached_compute = None
        
    def iteration_completed(self, engine, name):
        output = self._output_transform(engine.state.output)
        self.update(output)
        self._cached_compute = self.compute()
        engine.state.metrics[name] = self._cached_compute
        
    def completed(self, engine, name):
        engine.state.metrics[name] = self._cached_compute
        
    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED,
                                 self.iteration_completed, name)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)


class dice_loss(metric):
    '''
    Global Dice metric. Accumulates counts over the course of an epoch.
    
    target_class : integer or list
    mask_class : integer or list
    '''
    def __init__(self, target_class, target_index=0, mask_class=None,
                 output_transform=lambda x: x):
        super(dice_loss, self).__init__(output_transform)
        if not hasattr(target_class, '__len__'):
            self.target_class = [target_class]
        else:
            self.target_class = target_class
        if not hasattr(target_index, '__len__'):
            self.target_index = [target_index]
        else:
            self.target_index = target_index
        self.mask_class = mask_class
        self._smooth = 1
            
    def update(self, output):
        '''
        Expects integer or one-hot class labeling in y_true.
        Expects outputs in range [0, 1] in y_pred.
        
        Computes the soft dice loss considering all classes in target_class as
        one aggregate target class and ignoring all elements with ground truth
        classes in mask_class.
        '''
        
        # Get outputs.
        y_pred, y_true = output
        assert len(y_pred)==len(y_true)
        if y_true is None or len(y_true)==0:
            return
        y_pred = sum([y_pred[:,i:i+1] for i in self.target_index])
        
        # Targer variable must not require a gradient.
        assert(not y_true.requires_grad)
    
        # If needed, change ground truth from categorical to integer format.
        if y_true.ndimension() > y_pred.ndimension():
            y_true = torch.max(y_true, dim=1)[1]   # argmax
            
        # Flatten all inputs.
        y_true_f = y_true.view(-1).int()
        y_pred_f = y_pred.view(-1)
        
        # Aggregate target classes, mask out classes in mask_class.
        y_target = sum([y_true_f==t for t in self.target_class]).float()
        if self.mask_class is not None:
            mask_out = sum([y_true_f==t for t in self.mask_class])
            idxs = (mask_out==0).nonzero()
            y_target = y_target[idxs]
            y_pred_f = y_pred_f[idxs]
        
        # Accumulate dice counts.
        self._intersection += torch.sum(y_target * y_pred_f)
        self._y_target_sum += torch.sum(y_target)
        self._y_pred_sum   += torch.sum(y_pred)
        
    def compute(self):
        dice_val = -(2.*self._intersection+self._smooth) / \
                    (self._y_target_sum+self._y_pred_sum+self._smooth)
        return dice_val
    
    def reset(self):
        self._intersection = 0.
        self._y_target_sum = 0.
        self._y_pred_sum   = 0.
    
    
class batchwise_loss_accumulator(metric):
    """
    Accumulates a loss batchwise, weighted by the size of each batch.
    The batch size is determined as the number of elements in the loss
    `target`.
    
    output_transform : function that converts engine output to a
        (`loss`, `target`) tuple, where `target` is the tensor of target
        values against which the loss is computed.
    """
    def __init__(self, output_transform=lambda x: x):
        super(batchwise_loss_accumulator, self).__init__(output_transform)
    
    def update(self, output):
        loss, target = output
        if target is None or len(target)==0:
            return
        self._count += len(target)
        self._total += loss*len(target)
        
    def compute(self):
        return self._total/max(1., float(self._count))
    
    def reset(self):
        self._count = 0
        self._total = 0
