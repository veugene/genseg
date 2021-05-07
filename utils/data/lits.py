from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array


def prepare_data_lits(path, masked_fraction=0, drop_masked=False,
                      data_augmentation_kwargs=None, split_seed=0,
                      fold=None, rng=None):
    """
    Convenience function to prepare LiTS data split into training and
    validation subsets.
    
    path (string) : Path of the h5py file containing the LiTS data.
    masked_fraction (float) : The fraction in [0, 1.] of cases in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit cases with "masked" segmentations.
    split_seed (int) : The random seed to determine tha validation and test
        set split.
    fold (int) : The fold in {0,1,2,3} for 4-fold cross-validation. If None,
        then validation and test sets are some fixed 10% of the data, each.
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
    cases = list(f['s'].keys())
    n_cases = len(cases)
    rng_split = np.random.RandomState(split_seed)
    rng_split.shuffle(cases)
    if fold is None:
        # Random 20% data split (10% validation, 10% testing).
        split = {'train': cases[:int(0.8*n_cases)],
                 'valid': cases[int(0.8*n_cases):int(0.9*n_cases)],
                 'test' : cases[int(0.9*n_cases):]}
    else:
        # 25% of the data for test, 10% for validation. The validation
        # subset is randomly sampled from the data that isn't in the test
        # fold.
        assert fold in [0, 1, 2, 3]
        indices = [int(x*n_cases) for x in [0.25, 0.5, 0.75, 1]]
        fold_cases = {0: cases[0         :indices[0]],
                      1: cases[indices[0]:indices[1]],
                      2: cases[indices[1]:indices[2]],
                      3: cases[indices[2]:indices[3]]}
        test = fold_cases[fold]
        not_test = sum([fold_cases[x] for x in {0,1,2,3}-{fold}], [])
        rng_valid = np.random.RandomState(fold)     # Different for each fold.
        idx_valid = rng_valid.permutation(len(not_test))[:int(0.1*n_cases)]
        split = {'train': [x for i, x in enumerate(not_test)
                           if i not in idx_valid],
                 'valid': [x for i, x in enumerate(not_test)
                           if i in idx_valid],
                 'test' : test}
    
    # Assemble volumes.
    volumes_s = {'train': [f['s'][k] for k in split['train']],
                 'valid': [f['s'][k] for k in split['valid']],
                 'test' : [f['s'][k] for k in split['test']]}
    volumes_m = {'train': [f['m'][k] for k in split['train']],
                 'valid': [f['m'][k] for k in split['valid']],
                 'test' : [f['m'][k] for k in split['test']]}
    volumes_h = [f['h'][k] for k in f['h']]
    h_histogram = [f['h_histogram'][k] for k in f['h']]
    
    # Volumes with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of slices that
    # are to be thus removed. All or none of the slices are selected for 
    # each volume.
    n_cases_train = len(volumes_m['train'])
    num_total_slices = sum([len(v) for v in volumes_m['train']])
    sizes = np.array([len(v) for v in volumes_m['train']])
    # Mask all volumes that have more slices than the number that should be
    # visible.
    masked_arr = np.argwhere(sizes > num_total_slices*(1-masked_fraction))
    masked = list(masked_arr.flatten())
    visible = []
    if len(masked)==n_cases_train and masked_fraction!=1:
        # All volumes are masked but some slices should be visible.
        # Select the volume with the fewest slices to be visible.
        idx = np.argmin(sizes)
        masked.remove(idx)
        visible = [idx]
    # Keep masking volumes until enough slices are masked out.
    num_masked_slices = sum([len(volumes_m['train'][i]) for i in masked])
    max_masked_slices = int(min(num_total_slices,
                                num_total_slices*masked_fraction+0.5))
    for i in rng.permutation(n_cases_train):
        if i in masked:
            continue    # Already masked.
        num_slices = len(volumes_m['train'][i])
        if num_slices+num_masked_slices >= max_masked_slices:
            continue    # Stop masking non-empty volumes (mask empty).
        masked.append(i)
        num_masked_slices += num_slices
    visible.extend([i for i in range(n_cases_train) if i not in masked])
    print("DEBUG: A total of {}/{} slices are labeled across {} "
          "volumes ({:.1f}%)."
          "".format(num_total_slices-num_masked_slices,
                    num_total_slices,
                    len(volumes_m['train'])-len(masked),
                    100*(1-num_masked_slices/float(num_total_slices))))
    
    # Collect data in array-like interfaces. For the training set, mask out
    # a `masked_fraction` of sick cases.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
    if drop_masked:
        # Remove all volumes indexed with `masked`.
        data['train']['s'] = multi_source_array(
            [volumes_s['train'][i] for i in visible])
        data['train']['m'] = multi_source_array(
            [volumes_m['train'][i] for i in visible])
    else:
        # Mask out the labels for volumes indexed with `masked` by 
        # setting the segmentations volume as an array of `None`, with length 
        # equal to the number of slices in the volume.
        data['train']['s'] = multi_source_array(volumes_s['train'])
        data['train']['m'] = multi_source_array(
            [m if i in visible else np.array([None]*len(m))
             for i, m in enumerate(volumes_m['train'])])
    data['valid']['s'] = multi_source_array(volumes_s['valid'])
    data['valid']['m'] = multi_source_array(volumes_m['valid'])
    data['test']['s']  = multi_source_array(volumes_s['test'])
    data['test']['m']  = multi_source_array(volumes_m['test'])
    
    # HACK: we may have a situation where the number of sick examples is 
    # greater than the number of healthy. In that case, we should duplicate 
    # the healthy set `m` times so that it has a bigger size than the sick set.
    #
    # HACK: all folds have healthy data but all folds use the exact same 
    # healthy data. The model code expects this data but it is not needed
    # to compute segmentation performance, so we don't care that it's the same.
    #
    # Preload and normalize the `h_histogram`.
    for key in ['train', 'valid', 'test']:
        len_s = sum([len(elem) for elem in volumes_s[key]])
        len_h = sum([len(elem) for elem in volumes_h])
        m = int(np.ceil(len_s / len_h))
        data[key]['h'] = multi_source_array(volumes_h*m)
        h_histogram_arr = np.array(multi_source_array(h_histogram*m))
        data[key]['h_histogram'] = h_histogram_arr/h_histogram_arr.sum()
    
    # Create Dataset objects.
    dataset = {'train': None, 'valid': None, 'test': None}
    dataset['train'] = LITSDataset(**data['train'],
                                   da_kwargs=data_augmentation_kwargs,
                                   rng=rng)
    dataset['valid'] = LITSDataset(**data['valid'], rng=rng)
    dataset['test']  = LITSDataset(**data['test'],  rng=rng)
    return dataset


class LITSDataset(Dataset):
    """
    Returns sick slices by index according to `__getitem__()` and
    simultaneously returns healthy slices by sampling with replacement
    according to `h_histogram` weights.
    """
    def __init__(self, h, s, m, h_histogram, da_kwargs=None, rng=None):
        super().__init__()
        self.h = h
        self.s = s
        self.m = m
        self.h_histogram = h_histogram
        self.da_kwargs = da_kwargs
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState()
    
    def __getitem__(self, idx):
        s = self.s[idx]
        m = self.m[idx]
        h_idx = self.rng.choice(range(len(self.h)),
                                replace=True,
                                p=self.h_histogram)
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


def collate_lits(batch):
    h, s, m = zip(*batch)
    h = torch.as_tensor(h)
    s = torch.as_tensor(s)
    return h, s, m