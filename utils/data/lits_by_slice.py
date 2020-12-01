from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import (delayed_view,
                             multi_source_array)


def prepare_data_lits(path, masked_fraction=0, drop_masked=False,
                      data_augmentation_kwargs=None, split_seed=0,
                      rng=None):
    """
    Convenience function to prepare LiTS data split into training and
    validation subsets.
    
    path (string) : Path of the h5py file containing the LiTS data.
    masked_fraction (float) : The fraction in [0, 1.] of cases in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit cases with "masked" segmentations.
    split_seed (int) : The random seed to determine tha validation and test
        set split.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: The rng passed for data preparation is used to determine which 
    labels to mask out (if any); if none is passed, the default uses a
    random seed of 0.
    
    Returns a dictionary of pytorch `Dataset` objects for 'train', 'valid',
    and 'test' folds which sample sick slices with replacement and 
    simultaneously sample  healthy slices without replacement, using sampling
    weights equal to their corresponding position histogram values.
    """
    
    f = h5py.File(path, 'r')
    
    # Random 20% split (10% validation, 10% testing).
    s = multi_source_array([f['s'][k] for k in f['s']],
                           rng=np.random.RandomState(split_seed), shuffle=True)
    m = multi_source_array([f['m'][k] for k in f['s']],
                           rng=np.random.RandomState(split_seed), shuffle=True)
    h = multi_source_array([f['h'][k] for k in f['h']])
    w = multi_source_array([f['h_histogram'][k] for k in f['h']])
    w = np.array(w)
    w = w/w.sum()   # weights sum to 1
    assert len(s)==len(m)
    n_slices = len(s)
    idx_v = int(0.8*n_slices)
    idx_t = int(0.9*n_slices)
    data = {}
    data['train'] = {'s': delayed_view(s, idx_max=idx_v),
                     'm': delayed_view(m, idx_max=idx_v),
                     'h': h,
                     'w': w}
    data['valid'] = {'s': delayed_view(s, idx_min=idx_v, idx_max=idx_t),
                     'm': delayed_view(m, idx_min=idx_v, idx_max=idx_t),
                     'h': h,
                     'w': w}
    data['test']  = {'s': delayed_view(s, idx_min=idx_t),
                     'm': delayed_view(m, idx_min=idx_t),
                     'h': h,
                     'w': w}
    
    # Slices with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of slices that
    # are to be thus removed.
    if rng is None:
        rng = np.random.RandomState()
    n = len(data['train']['s'])
    masked_indices = rng.permutation(n)[:int(masked_fraction*n)]
    if drop_masked:
        visible_indices = [i for i in range(n) if i not in masked_indices]
        data['train']['s'] = _subset(data['train']['s'], visible_indices)
        data['train']['m'] = _subset(data['train']['m'], visible_indices)
        masked_indices = None   # Pass None to the Dataset.
    
    # Create Dataset objects.
    dataset = {'train': None, 'valid': None, 'test': None}
    dataset['train'] = LITSDataset(**data['train'],
                                   masked_indices=masked_indices,
                                   da_kwargs=data_augmentation_kwargs,
                                   rng=rng)
    dataset['valid'] = LITSDataset(**data['valid'], rng=rng)
    dataset['test']  = LITSDataset(**data['test'],  rng=rng)
    return dataset


class LITSDataset(Dataset):
    """
    Returns sick slices by index according to `__getitem__()` and
    simultaneously returns healthy slices by sampling with replacement
    according to `w` weights.
    """
    def __init__(self, h, s, m, w,
                 masked_indices=None, da_kwargs=None, rng=None):
        super().__init__()
        self.h = h
        self.s = s
        self.m = m
        self.w = w
        self.masked_indices = masked_indices
        self.da_kwargs = da_kwargs
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()
    
    def __getitem__(self, idx):
        s = self.s[idx]
        m = self.m[idx]
        if self.masked_indices is not None and idx in self.masked_indices:
            m = None
        h_idx = self.rng.choice(range(len(self.h)),
                                replace=True,
                                p=self.w)
        h = self.h[h_idx]
        
        # Add channel dim: h, s may have 2 or 3 dims, depending on if liver
        # is cropped.
        if h.ndim==2:
            h = np.expand_dims(h, 0)
        if s.ndim==2:
            s = np.expand_dims(s, 0)
        if m is not None:
            m = np.expand_dims(m, 0)    # (256, 256) -> (1, 256, 256)

        # Float.
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        
        # Data augmentation.
        if self.da_kwargs is not None:
            h = image_random_transform(h, **self.da_kwargs)
            s = image_random_transform(s, m, **self.da_kwargs)
            if m is not None:
                s, m = s
        
        # Remove distant outlier intensities.
        if h is not None:
            h = np.clip(h, -1., 1.)
        if s is not None:
            s = np.clip(s, -1., 1.)
        
        return h, s, m
    
    def __len__(self):
        return len(self.s)


class _subset(object):
    def __init__(self, arr, indices):
        self.arr = arr
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.arr[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def collate_lits(batch):
    h, s, m = zip(*batch)
    h = torch.as_tensor(h)
    s = torch.as_tensor(s)
    return h, s, m