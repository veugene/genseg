from collections import OrderedDict

import h5py
import numpy as np
from skimage import filters
from skimage import transform

from data_tools.wrap import multi_source_array
from data_tools.data_augmentation import image_random_transform

 
def prepare_data_ddsm(path, masked_fraction=0, drop_masked=False, rng=None):
    """
    Convenience function to prepare DDSM data as multi_source_array objects,
    split into training and validation subsets.
    
    path (string) : Path of the h5py file containing the DDSM data.
    masked_fraction (float) : The fraction in [0, 1.] of cases in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit cases with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation split. The rng passed for data preparation is used to determine
    which labels to mask out (if any); if none is passed, the default uses a
    random seed of 0.
    
    Returns dictionary: healthy slices, sick slices, and segmentations for 
    the training, validation, and testing subsets.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if masked_fraction < 0 or masked_fraction > 1:
        raise ValueError("`masked_fraction` must be in [0, 1].")
    
    # Random 20% data split for validation, random 20% for testing.
    # 
    # First, split out a random sample of the sick patients into a
    # validation/test set. Any sick patients in the validation/test set that 
    # reoccur in the healthy set are placed in the validation/test set for 
    # the healthy set, as well. The remainder of the healthy validation/test
    # set is completed with a random sample.
    case_id_h = {}
    case_id_s = {}
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    rnd_state = np.random.RandomState(0)
    indices = {'s': rnd_state.permutation(len(h5py_file['sick'].keys())),
               'h': rnd_state.permutation(len(h5py_file['healthy'].keys()))}
    def get(keys, indices): # Index into keys object.
        return [k for i, k in enumerate(keys) if i in indices]
    def split(n_cases):
        indices_s_split = indices['s'][:n_cases]
        case_id_s_split = get(h5py_file['sick'].keys(), indices_s_split)
        indices_s_and_h = [i for i, k in enumerate(h5py_file['healthy'].keys())
                           if k in case_id_s_split]
        indices_h_only  = [i for i, k in enumerate(h5py_file['healthy'].keys())
                           if i not in indices_s_and_h]
        n_random = n_cases-len(indices_s_and_h)
        indices_h_split = indices_s_and_h+indices_h_only[:n_random]
        case_id_h_split = get(h5py_file['healthy'].keys(), indices_h_split)
        indices['s'] = [i for i in indices['s'] if i not in indices_s_split]
        indices['h'] = [i for i in indices['h'] if i not in indices_h_split]
        return case_id_s_split, case_id_h_split
    n_valid = int(0.2*len(indices['s']))
    case_id_s['valid'], case_id_h['valid'] = split(n_valid)
    case_id_s['test'],  case_id_h['test']  = split(n_valid)
    
    # Remaining cases go in the training set. At this point, indices for
    # cases in the validation and test sets have been removed.
    case_id_s['train'] = get(h5py_file['sick'].keys(),    indices['s'])
    case_id_h['train'] = get(h5py_file['healthy'].keys(), indices['h'])
    indices = None  # Clear remaining indices.
    
    # Assemble cases with corresponding segmentations; split train/valid.
    cases_h = {'train': [], 'valid': [], 'test': []}
    cases_s = {'train': [], 'valid': [], 'test': []}
    cases_m = {'train': [], 'valid': [], 'test': []}
    for split in ['train', 'valid', 'test']:
        for case_id in case_id_s[split]:
            f = h5py_file['sick'][case_id]
            for view in f.keys():
                for key in filter(lambda x:x.startswith('abnormality'),
                                  f[view].keys()):
                    cases_s[split].append(f[view][key]['cropped_image'])
                    cases_m[split].append(f[view][key]['cropped_mask'])
        for case_id in case_id_h[split]:
            f = h5py_file['healthy'][case_id]
            for view in f.keys():
                cases_h[split].append(f[view]['patch'])

    # Cases with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of cases that
    # are to be thus removed.
    num_cases_total  = len(cases_m['train'])
    num_cases_masked = int(min(num_cases_total,
                               num_cases_total*masked_fraction+0.5))
    masked_indices = rng.permutation(num_cases_total)[:num_cases_masked]
    print("DEBUG: A total of {}/{} images are labeled."
          "".format(num_cases_total-num_cases_masked, num_cases_total))
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for cases indexed with `masked_indices` by 
    # setting the segmentation as an array of `None`.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all cases indexed with `masked_indices`.
    cases_s_train = []
    cases_m_train = []
    for i in range(num_cases_total):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            cases_m_train.append(None)
        else:
            # Keep.
            cases_m_train.append(cases_m['train'][i])
        cases_s_train.append(cases_s['train'][i])
    cases_s['train'] = cases_s_train
    cases_m['train'] = cases_m_train
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test',  OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in cases_h[key]])
        len_s = sum([len(elem) for elem in cases_s[key]])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['h'] = _list(cases_h[key]*m, np.uint16)
        data[key]['s'] = _list(cases_s[key], np.uint16)
        data[key]['m'] = _list(cases_m[key], np.uint8)
    return data


class _list(object):
    def __init__(self, indexable, dtype):
        self._items = indexable
        self.dtype = dtype
    def __getitem__(self, idx):
        elem = self._items[idx]
        if elem is not None:
            elem = elem[...]
        return elem
    def __len__(self):
        return len(self._items)


def preprocessor_ddsm(resize_to, crop_to=None, data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for DDSM data.
    
    resize_to : An int defining the spatial size to resize all inputs to.
    crop_to : An int defining the spatial size to crop all inputs to,
        after resizing and data augmentation (if any). Crops are centered.
        Used for processing images for validation.
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    """
    def process_element(inputs):
        h, s, m = inputs

        # Float, rescale, center.
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        h = h/2**15 - 1
        s = s/2**15 - 1
        
        # Resize.
        h = resize(h, size=(resize_to, resize_to), order=1)
        s = resize(s, size=(resize_to, resize_to), order=1)
        if m is not None:
            m = resize(m, size=(resize_to, resize_to), order=0)
        
        # Expand dims.
        h = np.expand_dims(h, 0)
        s = np.expand_dims(s, 0)
        if m is not None:
            m = np.expand_dims(m, 0)
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
            h = image_random_transform(h, **data_augmentation_kwargs,
                                       n_warp_threads=1)
            _ = image_random_transform(s, m, **data_augmentation_kwargs,
                                       n_warp_threads=1)
            if m is not None:
                s, m = _
            else:
                s = _
        
        # Crop images (centered) -- for validation.
        if crop_to is not None:
            assert np.all(h.shape==s.shape)
            assert np.all(h.shape==m.shape)
            x, y = np.subtract(h.shape[-2:], crop_to)//2
            h = h[:, x:x+crop_to, y:y+crop_to]
            s = s[:, x:x+crop_to, y:y+crop_to]
            m = m[:, x:x+crop_to, y:y+crop_to]
        
        return h, s, m
    
    def process_batch(batch):
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch


def resize(image, size, order=1):
    image = image.copy()
    out_image = np.zeros(image.shape[:-2]+size, dtype=image.dtype)
    for idx in np.ndindex(image.shape[:-2]):
        if any([a<b for a,b in zip(size, image.shape[-2:])]) and order>0:
            # Downscaling - smooth, first.
            s = [(1-float(a)/b)/2. if a<b else 0
                 for a,b in zip(size, image.shape[-2:])]
            image[idx] = filters.gaussian(image[idx], sigma=s)
        out_image[idx] = transform.resize(image[idx],
                                          output_shape=size,
                                          mode='constant',
                                          order=order,
                                          cval=0,
                                          clip=False,
                                          preserve_range=True)
    return out_image