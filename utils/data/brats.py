from collections import OrderedDict

import h5py
import numpy as np

from data_tools.wrap import multi_source_array
from data_tools.data_augmentation import image_random_transform

 
def prepare_data_brats17(path_hgg, path_lgg,
                         masked_fraction=0, drop_masked=False,
                         rng=None):
    # Random 20% data split.
    rnd_state = np.random.RandomState(0)
    hgg_indices = np.arange(0, 210)
    lgg_indices = np.arange(0, 75)
    rnd_state.shuffle(hgg_indices)
    rnd_state.shuffle(lgg_indices)
    hgg_val = hgg_indices[0:int(0.2*210)]
    lgg_val = lgg_indices[0:int(0.2*75)]
    validation_indices = {'hgg': hgg_val, 'lgg': lgg_val}
    return _prepare_data_brats(path_hgg, path_lgg,
                               masked_fraction=masked_fraction,
                               validation_indices=validation_indices,
                               drop_masked=drop_masked,
                               rng=rng)

def prepare_data_brats13s(path_hgg, path_lgg,
                          masked_fraction=0, drop_masked=False,
                          rng=None):
    rnd_state = np.random.RandomState(0)
    hgg_indices = np.arange(0, 25)
    lgg_indices = np.arange(0, 25)
    rnd_state.shuffle(hgg_indices)
    rnd_state.shuffle(lgg_indices)
    hgg_val = hgg_indices[0:int(0.2*25)]
    lgg_val = lgg_indices[0:int(0.2*25)]
    validation_indices = {'hgg': hgg_val, 'lgg': lgg_val}
    return _prepare_data_brats(path_hgg, path_lgg,
                               masked_fraction=masked_fraction,
                               validation_indices=validation_indices,
                               drop_masked=drop_masked,
                               rng=rng)

def _prepare_data_brats(path_hgg, path_lgg, validation_indices,
                        masked_fraction=0, drop_masked=False,
                        rng=None):
    """
    Convenience function to prepare brats data as multi_source_array objects,
    split into training and validation subsets.
    
    path_hgg (string) : Path of the h5py file containing the HGG data.
    path_lgg (string) : Path of the h5py file containing the LGG data.
    masked_fraction (float) : The fraction in [0, 1.] of volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    rng (numpy RandomState) : Random number generator.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation split. The rng passed for data preparation is used to determine
    which labels to mask out (if any); if none is passed, the default uses a
    random seed of 0.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    if rng is None:
        rng = np.random.RandomState(0)
    if masked_fraction < 0 or masked_fraction > 1:
        raise ValueError("`masked_fraction` must be in [0, 1].")
        
    # The rest is training data.
    with h5py.File(path_lgg, 'r') as f:
        num_lgg = len(f.keys())
    with h5py.File(path_hgg, 'r') as f:
        num_hgg = len(f.keys())
    
    # Check if any of the validation indices exceeds
    # the length of the # of examples.
    max_hgg_idx = max(validation_indices['hgg'])
    if max_hgg_idx >= num_hgg:
        raise Exception("Max validation index is {} but len(hgg) is {}".
                        format(max_hgg_idx, num_hgg))
    max_lgg_idx = max(validation_indices['lgg'])
    if max_lgg_idx >= num_lgg:
        raise Exception("Max validation index is {} but len(lgg) is {}".
                        format(max_lgg_idx, num_lgg))
    
    # Assemble volumes and corresponding segmentations; split train/valid.
    path = OrderedDict((('hgg', path_hgg), ('lgg', path_lgg)))
    volumes_h = {'train': [], 'valid': []}
    volumes_s = {'train': [], 'valid': []}
    volumes_m = {'train': [], 'valid': []}
    indices_h = {'train': [], 'valid': []}
    indices_s = {'train': [], 'valid': []}
    for key, path in path.items():
        try:
            h5py_file = h5py.File(path, mode='r')
        except:
            print("Failed to open data: {}".format(path))
            raise
        for idx, case_id in enumerate(h5py_file.keys()):   # Per patient.
            f = h5py_file[case_id]
            if idx in validation_indices[key]:
                split = 'valid'
            else:
                split = 'train'
            volumes_h[split].append(f['healthy'])
            volumes_s[split].append(f['sick'])
            volumes_m[split].append(f['segmentation'])
            indices_h[split].append(f['h_indices'])
            indices_s[split].append(f['s_indices'])

    # Volumes with these indices will either be dropped from the training
    # set or have their segmentations set to None.
    # 
    # The `masked_fraction` determines the maximal fraction of slices that
    # are to be thus removed. All or none of the slices are selected for 
    # each volume.
    masked_indices = []
    num_total_slices = sum([len(v) for v in volumes_m['train']])
    num_masked_slices = 0
    max_masked_slices = int(min(num_total_slices,
                                num_total_slices*masked_fraction+0.5))
    for i in rng.permutation(len(volumes_m['train'])):
        num_slices = len(volumes_m['train'][i])
        if num_slices>0 and num_masked_slices >= max_masked_slices:
            continue    # Stop masking non-empty volumes (mask empty).
        if num_slices+num_masked_slices >= max_masked_slices:
            continue    # Stop masking non-empty volumes (mask empty).
        masked_indices.append(i)
        num_masked_slices += num_slices
    print("DEBUG: A total of {}/{} slices are labeled across {} "
          "volumes ({:.1f}%)."
          "".format(num_total_slices-num_masked_slices,
                    num_total_slices,
                    len(volumes_m['train'])-len(masked_indices),
                    100*(1-num_masked_slices/float(num_total_slices))))
    
    # Apply masking in one of two ways.
    # 
    # 1. Mask out the labels for volumes indexed with `masked_indices` by 
    # setting the segmentations volume as an array of `None`, with length 
    # equal to the number of slices in the volume.
    # 
    # OR if `drop_masked` is True:
    # 
    # 2. Remove all volumes indexed with `masked_indices`.
    volumes_h_train = []
    volumes_s_train = []
    volumes_m_train = []
    indices_h_train = []
    indices_s_train = []
    for i in range(len(volumes_m['train'])):
        if i in masked_indices:
            # Mask out or drop.
            if drop_masked:
                continue    # Drop.
            volumes_m_train.append(np.array([None]*len(volumes_m['train'][i])))
        else:
            # Keep.
            volumes_m_train.append(volumes_m['train'][i])
        volumes_h_train.append(volumes_h['train'][i])
        volumes_s_train.append(volumes_s['train'][i])
        indices_h_train.append(indices_h['train'][i])
        indices_s_train.append(indices_s['train'][i])
    volumes_h['train'] = volumes_h_train
    volumes_s['train'] = volumes_s_train
    volumes_m['train'] = volumes_m_train
    indices_h['train'] = indices_h_train
    indices_s['train'] = indices_s_train
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict())])
    for key in data.keys():
        # HACK: we may have a situation where the number of sick examples
        # is greater than the number of healthy. In that case, we should
        # duplicate the healthy set M times so that it has a bigger size
        # than the sick set.
        m = 1
        len_h = sum([len(elem) for elem in volumes_h[key]])
        len_s = sum([len(elem) for elem in volumes_s[key]])
        if len_h < len_s:
            m = int(np.ceil(len_s / len_h))
        data[key]['h']  = multi_source_array(volumes_h[key]*m, no_shape=True)
        data[key]['s']  = multi_source_array(volumes_s[key],   no_shape=True)
        data[key]['m']  = multi_source_array(volumes_m[key],   no_shape=True)
        data[key]['hi'] = multi_source_array(indices_h[key]*m, no_shape=True)
        data[key]['si'] = multi_source_array(indices_s[key],   no_shape=True)
    return data


def preprocessor_brats(data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    """
        
    def process_element(inputs, max_shape):
        h, s, m, hi, si = inputs
        
        # Center slices onto empty buffers with a size equal to the largest.
        elem = []
        for im in [h, s, m]:
            if im is None:
                elem.append(None)
                continue
            elem_im = np.zeros((len(im),)+max_shape, dtype=np.float32)
            elem_im[...] = np.inf   # To be replaced with background intensity.
            im_shape = np.shape(im)[1:]
            offset = np.subtract(max_shape, im_shape)//2
            index_slices = [slice(None, None),
                            slice(offset[0], offset[0]+im_shape[0]),
                            slice(offset[1], offset[1]+im_shape[1])]
            elem_im[tuple(index_slices)] = im[...]
            elem.append(elem_im)
        h, s, m = elem
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
            if h is not None:
                h = image_random_transform(h, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if s is not None:
                _ = image_random_transform(s, m, **data_augmentation_kwargs,
                                           n_warp_threads=1)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
                        
        # Set background intensity.
        for im, im_orig in zip([h, s, m], inputs[:-2]):
            if im is None:
                continue
            # HACK: Assuming corner pixel is always outside of the brain.
            background = im_orig[0,0,0]
            im[im==np.inf] = background
                    
        # Remove distant outlier intensities.
        hs = []
        for im in [h, s]:
            if im is None:
                hs.append(None)
                continue
            im = np.clip(im, -1., 1.)
            hs.append(im)
        h, s = hs
        
        # Set dtype (all output buffers are float32 to support inf).
        elem = []
        for im, im_orig in zip([h, s, m], inputs[:-2]):
            if im is None:
                elem.append(None)
                continue
            im = im.astype(im_orig.dtype)
            elem.append(im)
        
        # Append indices.
        elem.extend(inputs[-2:])
            
        return elem
        
    def process_batch(batch):
        # Find the largest slice.
        max_shape = (0,0)
        for b in batch[:-2]:
            for im in b:
                if im is None:
                    continue
                im_shape = np.shape(im)[1:]
                max_shape = (max(im_shape[0], max_shape[0]),
                             max(im_shape[1], max_shape[1]))
                
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch], max_shape=max_shape)
            elements.append(elem)
        out_batch = list(zip(*elements))
        
        return out_batch
    
    return process_batch
