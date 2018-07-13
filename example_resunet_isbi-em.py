from __future__ import (print_function,
                        division)
from builtins import input
from collections import OrderedDict
import sys

import numpy as np
import tifffile
import torch
import ignite
from ignite.engines import (Events,
                            Trainer,
                            Evaluator)
from torch.autograd import Variable

from architectures.revnet import dilated_fcn_hybrid
from architectures.blocks import reversible_basic_block
from data_tools.io import data_flow
from data_tools.data_augmentation import image_stack_random_transform

from utils.ignite import (progress_report,
                          metrics_handler)
from utils.metrics import (dice_loss,
                           accuracy)
                          

'''
Settings.
'''
model_kwargs = OrderedDict((
    ('in_channels', 1),
    ('num_blocks', 6),
    ('filters', [8,16,32,64,64,64,64,64,128,128,64,32,16,8]),
    ('num_downscale', 3),
    ('dilation', [1,2,4,8,16,1]),
    ('patch_size', None),
    ('short_skip', True),
    ('long_skip', True),
    ('long_skip_merge_mode', 'sum'),
    ('upsample_mode', 'repeat'),
    ('dropout', 0.),
    ('norm_kwargs', None),
    ('init', 'kaiming_normal'),
    ('nonlinearity', 'ReLU'),
    ('block_type', reversible_basic_block),
    ('num_classes', 1),
    ('ndim', 2),
    ('verbose', True),
    ))
batch_size = 4


'''
Set paths.
'''
data_path = "/tmp/datasets/isbi_2012_em/"
ds_path = {'train-volume': data_path + "train-volume.tif",
           'train-labels': data_path + "train-labels.tif",
           'test-volume': data_path + "test-volume.tif"}


if __name__=='__main__':
    '''
    Prepare data -- load, standardize, add channel dim., shuffle, split.
    '''
    # Load
    X = tifffile.imread(ds_path['train-volume']).astype(np.float32)
    Y = tifffile.imread(ds_path['train-labels']).astype(np.int64)
    Y[Y==255] = 1
    # Standardize each sample individually (mean center, variance normalize)
    mean = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).mean(axis=-1)[:,None,None]
    std  = X.reshape(X.shape[0],
                     np.prod(X.shape[1:])).std(axis=-1)[:,None,None]
    X = (X-mean)/std
    #X /= 255.
    # Add channel dim
    X = np.expand_dims(X, axis=1)
    Y = np.expand_dims(Y, axis=1)
    # Shuffle
    R = np.random.permutation(len(X))
    X = X[R]
    Y = Y[R]
    Y = 1-Y
    # Split (26 training, 4 validation)
    X_train = X[:26]
    Y_train = Y[:26]
    X_valid = X[26:]
    Y_valid = Y[26:]
    # Prepare data augmentation and data loaders.
    def preprocessor_train(batch):
        b0, b1 = batch
        b0, b1 = image_stack_random_transform(x=b0, y=b1,
                                              rotation_range=25,
                                              shear_range=0.41,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              spline_warp=True,
                                              warp_sigma=10,
                                              warp_grid_size=3,
                                              fill_mode='reflect')
        b1 = np.array(b1, dtype=np.int64)
        return b0, b1
    loader_train = data_flow(data=[X_train, Y_train],
                             batch_size=batch_size,
                             preprocessor=preprocessor_train,
                             sample_random=True)
    loader_valid = data_flow(data=[X_valid, Y_valid],
                             batch_size=batch_size)
    
    '''
    Prepare model.
    '''
    torch.backends.cudnn.benchmark = True
    model = dilated_fcn_hybrid(**model_kwargs)
    model.cuda()
    optimizer = torch.optim.RMSprop(params=model.parameters(),
                                    lr=0.001, alpha=0.9,
                                    weight_decay=1e-4)
    loss_function = dice_loss(target_class=1,
                              target_index=0).cuda()
    
    '''
    Set up metrics.
    '''
    metrics = {}
    for key in ['train', 'valid']:
        metrics[key] = metrics_handler({'dice_loss': dice_loss(target_class=1,
                                                               target_index=0),
                                        'accuracy': accuracy})
    
    '''
    Set up training and evaluation functions.
    '''
    def prepare_batch(batch):
        b0, b1 = batch
        b0 = Variable(torch.from_numpy(np.array(b0))).cuda()
        b1 = Variable(torch.from_numpy(np.array(b1))).cuda()
        return b0, b1
    
    def training_function(engine, batch):
        batch = prepare_batch(batch)
        model.train()
        optimizer.zero_grad()
        output = model(batch[0])
        loss = loss_function(output, batch[1])
        loss.backward()
        optimizer.step()
        return loss.item(), metrics['train'](output.detach(), batch[1])
    trainer = Trainer(training_function)
        
    def validation_function(engine, batch):
        batch = prepare_batch(batch)
        model.eval()
        with torch.no_grad():
            output = model(batch[0])
            loss = loss_function(output, batch[1])
            correct = batch[1].view_as(output).sum()
            accuracy = correct.float()/batch[1].nelement()
        return loss.item(), metrics['valid'](output, batch[1])
    evaluator = Evaluator(validation_function)
    
    '''
    Set up logging to screen.
    '''
    def epoch_length(dataset):
        return len(dataset)//batch_size + int(len(dataset)%batch_size>0)
    progress_train = progress_report(epoch_length=epoch_length(X_train))
    progress_valid = progress_report(epoch_length=epoch_length(X_valid),
                                     prefix="val",
                                     progress_bar=False)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, progress_train)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _ : evaluator.run(loader_valid))
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, progress_valid)
    
    '''
    Train.
    '''
    trainer.run(loader_train, max_epochs=20)
    
    '''
    Predict and save images.
    '''
    X_valid_tensor = torch.from_numpy(X_valid)
    X_valid_tensor = X_valid_tensor.cuda()
    X_valid_var = torch.autograd.Variable(X_valid_tensor, requires_grad=False)
    prediction = model(X_valid_var)
    #prediction = prepare_batch(next(iter(loader_train)))[0]
    prediction = np.squeeze(prediction.data.cpu().numpy())
    for i, im in enumerate(prediction):
        from scipy.misc import imsave
        imsave("{}.png".format(i), im)
