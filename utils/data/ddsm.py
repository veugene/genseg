from collections import OrderedDict

import h5py
import numpy as np
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
    the training and validation subsets.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if masked_fraction < 0 or masked_fraction > 1:
        raise ValueError("`masked_fraction` must be in [0, 1].")
    
    # Random 20% data split for validation.
    # 
    # First, split out a random sample of the sick patients into a validation
    # set. Any sick patients in the validation set that reoccur in the
    # healthy set are placed in the validation set for the healthy set, as 
    # well. The remainder of the healthy validation set is completed with 
    # a random sample.
    case_id_h = {}
    case_id_s = {}
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    case_id_s_set = set(h5py_file['sick'].keys())
    case_id_h_set = set(h5py_file['healthy'].keys())
    case_id_i_set = cases_s.intersection(cases_h)
    case_id_h_unique = [case in h5py_file['healthy'].keys()
                        if case in case_id_h_set.difference(case_id_i_set)]
    rnd_state = np.random.RandomState(0)
    indices_s = rnd_state.permutation(len(case_id_s_set))
    indices_h = rnd_state.permutation(len(case_id_h_unique))
    n_valid = int(0.2*len(indices_s))   # Determined by sick cases.
    assert n_valid > len(case_id_h_set)
    indices_s_train, indices_s_valid = np.split(indices_s, [n_valid])
    case_id_s['valid'] = h5py_file['sick'].keys()[indices_sick_valid]
    case_id_s['train'] = h5py_file['sick'].keys()[indices_sick_train]
    case_id_h['valid'] = ( list(case_id_i_set)
                          +case_id_h_unique[indices_h[:n_valid]])
    
    # Assemble cases with corresponding segmentations; split train/valid.
    cases_h = {'train': [], 'valid': []}
    cases_s = {'train': [], 'valid': []}
    cases_m = {'train': [], 'valid': []}
    for split in ['train', 'valid']:
        for case_id in case_id_s[split]:
            f = h5py_file['sick'][case_id]
            for key in filter(lambda x:x.startswith('abnormality'), f.keys()):
                cases_s[split].append(f[key]['cropped_image'])
                cases_m[split].append(f[key]['cropped_mask'])
        for case_id in case_id_h[split]:
            cases_h[split].append(h5py_file['healthy'][case_id]['patch'])

    # Cases with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of cases that
    # are to be thus removed.
    num_cases_total  = len(cases_m['train'])
    num_cases_masked = int(min(num_cases_total,
                               num_cases_total*masked_fraction+0.5))
    masked_indices = rng.permutation(num_cases_total)[:num_cases_masked]
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for cases indexed with `masked_indices` by 
    # setting the segmentation as an array of `None`.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all cases indexed with `masked_indices`.
    cases_h_train = []
    cases_s_train = []
    cases_m_train = []
    for i in range(num_cases_total):
        if drop_masked:
            # Drop.
            continue
        elif i not in masked_indices:
            # Keep.
            cases_m_train.append(cases_m['train'][i])
        else:
            # Mask out.
            cases_m_train.append(np.array([None]))
        cases_h_train.append(cases_h['train'][i])
        cases_s_train.append(cases_s['train'][i])
    cases_h['train'] = cases_h_train
    cases_s['train'] = cases_s_train
    cases_m['train'] = cases_m_train
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict())])
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
        data[key]['h']  = multi_source_array(cases_h[key]*m, no_shape=True)
        data[key]['s']  = multi_source_array(cases_s[key],   no_shape=True)
        data[key]['m']  = multi_source_array(cases_m[key],   no_shape=True)
    return data


def preprocessor_ddsm(output_size, data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for DDSM data.
    
    output_size : A tuple defining the spatial size to resize all inputs to.
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    """
    def process_batch(batch):
        h, s, m = batch
        h, s, m = resize(h), resize(s), resize(m, interp='constant')
        if data_augmentation_kwargs is not None:
            h    = image_random_transform(h, **data_augmentation_kwargs,
                                          n_warp_threads=1)
            s, m = image_random_transform(s, m, **data_augmentation_kwargs,
                                          n_warp_threads=1)
        return h, s, m
    return process_batch


def resize(image_list, size, interp='bilinear'):
    """
    Resize each image in a list.
    """
    out = []
    for image in image_list:
        out_image = np.zeros(image.shape[:-2]+size, dtype=image.dtype)
        for idx in np.ndindex(image.shape[:-2]):
            out_image[idx] = transform.resize(image[idx],
                                              output_shape=size,
                                              mode=interp,
                                              cval=0,
                                              clip=False,
                                              preserve_range=True)
        out.append(out_image)
    return out