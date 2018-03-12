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
        self.target_class = target_class
        self.target_index = target_index
        self.mask_class = mask_class
        self._dice_loss = _dice_loss(target_class, mask_class)

    def forward(self, y_pred, y_true):
        idx = self.target_index
        return self._dice_loss(y_pred[:,idx:(idx+1)], y_true)
