from fcn_maker.loss import dice_loss

def accuracy(output, target):
    if output.size(1) > 1:
        compare = output.max(dim=1, keepdim=True)[0].long()
    else:
        compare = output.round().long()
    return compare.eq(target).float().sum() / target.nelement() 
