from __future__ import (print_function,
                        division)
import torch
from fcn_maker.loss import dice_loss as _dice_loss


def accuracy(output, target):
    if output.size(1) > 1:
        compare = output.max(dim=1, keepdim=True)[0].long()
    else:
        compare = output.round().long()
    return compare.eq(target).float().sum() / target.nelement() 


class dice_loss(torch.nn.Module):
    def __init__(self, target_class, target_index, mask_class=None):
        """
        target_class : the integer label in the ground truth
        target_index : the index into the output feature map corresponding
          to `target_class`.
        """
        super(dice_loss, self).__init__()
        if not hasattr(target_class, '__len__'):
            self.target_class = [target_class]
        else:
            self.target_class = target_class
        if not hasattr(target_index, '__len__'):
            self.target_index = [target_index]
        else:
            self.target_index = target_index
        self.mask_class = mask_class
        self._dice_loss = _dice_loss(target_class, mask_class)

    def forward(self, y_pred, y_true):
        y_pred_indexed = sum([y_pred[:,i:i+1] for i in self.target_index])
        return self._dice_loss(y_pred_indexed.contiguous(), y_true)


class global_dice(dice_loss):
    '''
    Global Dice metric. Accumulates counts over the course of an epoch.
    '''
    def __init__(self, target_class, target_index, mask_class=None,
                 epoch_length=None):
        super(global_dice, self).__init__(target_class=target_class,
                                          target_index=target_index,
                                          mask_class=mask_class)
        self.epoch_length = epoch_length
        self._smooth = 1
        self._iteration = 0
        self._intersection = 0
        self._y_target_sum = 0
        self._y_pred_sum = 0
            
    def _dice_loss(self, y_pred, y_true):
        '''
        Expects integer or one-hot class labeling in y_true.
        Expects outputs in range [0, 1] in y_pred.
        
        Computes the soft dice loss considering all classes in target_class as one
        aggregate target class and ignoring all elements with ground truth classes
        in mask_class.
        
        target_class : integer or list
        mask_class : integer or list
        '''
        
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
        if self.epoch_length is not None:
            if self._iteration==self.epoch_length:
                self.reset_counts()
        self._intersection += torch.sum(y_target * y_pred_f).item()
        self._y_target_sum += torch.sum(y_target).item()
        self._y_pred_sum += torch.sum(y_pred).item()
        self._iteration += 1
        
        # Compute dice.
        dice_val = -(2.*self._intersection+self._smooth) / \
                    (self._y_target_sum+self._y_pred_sum+self._smooth)
        return torch.FloatTensor([dice_val])
    
    def reset_counts(self, *args, **kwargs):
        self._intersection = 0
        self._y_target_sum = 0
        self._y_pred_sum = 0
