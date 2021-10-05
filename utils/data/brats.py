from collections import OrderedDict

import h5py
import numpy as np
from scipy import ndimage

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array

 
def prepare_data_brats17(path_hgg, path_lgg,
                         masked_fraction=0, drop_masked=False,
                         rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    hgg_indices = np.arange(0, 210)
    lgg_indices = np.arange(0, 75)
    rnd_state.shuffle(hgg_indices)
    rnd_state.shuffle(lgg_indices)
    hgg_val  = hgg_indices[0:21]
    lgg_val  = lgg_indices[0:7]
    hgg_test = hgg_indices[21:42]
    lgg_test = lgg_indices[7:15]
    validation_indices = {'hgg': hgg_val,  'lgg': lgg_val}
    testing_indices    = {'hgg': hgg_test, 'lgg': lgg_test}
    return _prepare_data_brats(path_hgg, path_lgg,
                               masked_fraction=masked_fraction,
                               validation_indices=validation_indices,
                               testing_indices=testing_indices,
                               drop_masked=drop_masked,
                               rng=rng)

def prepare_data_brats13s(path_hgg, path_lgg,
                          masked_fraction=0, drop_masked=False,
                          rng=None):
    # Random 28% data split (12% validation, 16% testing).
    rnd_state = np.random.RandomState(0)
    hgg_indices = np.arange(0, 25)
    lgg_indices = np.arange(0, 25)
    rnd_state.shuffle(hgg_indices)
    rnd_state.shuffle(lgg_indices)
    hgg_val  = hgg_indices[0:3]
    lgg_val  = lgg_indices[0:3]
    hgg_test = hgg_indices[3:7]
    lgg_test = lgg_indices[3:7]
    validation_indices = {'hgg': hgg_val,  'lgg': lgg_val}
    testing_indices    = {'hgg': hgg_test, 'lgg': lgg_test}
    return _prepare_data_brats(path_hgg, path_lgg,
                               masked_fraction=masked_fraction,
                               validation_indices=validation_indices,
                               testing_indices=testing_indices,
                               drop_masked=drop_masked,
                               rng=rng)

def _prepare_data_brats(path_hgg, path_lgg, validation_indices,
                        testing_indices, masked_fraction=0, drop_masked=False,
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
    validation/testing split. The rng passed for data preparation is used to 
    determine which labels to mask out (if any); if none is passed, the default
    uses a random seed of 0.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    if rng is None:
        rng = np.random.RandomState(0)
    if masked_fraction < 0 or masked_fraction > 1:
        raise ValueError("`masked_fraction` must be in [0, 1].")
    
    # Assemble volumes and corresponding segmentations; split train/valid/test.
    path = OrderedDict((('hgg', path_hgg), ('lgg', path_lgg)))
    volumes_h = {'train': [], 'valid': [], 'test': []}
    volumes_s = {'train': [], 'valid': [], 'test': []}
    volumes_m = {'train': [], 'valid': [], 'test': []}
    indices_h = {'train': [], 'valid': [], 'test': []}
    indices_s = {'train': [], 'valid': [], 'test': []}
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
            elif idx in testing_indices[key]:
                split = 'test'
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
                        ('valid', OrderedDict()),
                        ('test', OrderedDict())])
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
        data[key]['h']  = multi_source_array(volumes_h[key]*m)
        data[key]['s']  = multi_source_array(volumes_s[key])
        data[key]['m']  = multi_source_array(volumes_m[key])
        data[key]['hi'] = multi_source_array(indices_h[key]*m)
        data[key]['si'] = multi_source_array(indices_s[key])
    return data


def preprocessor_brats(data_augmentation_kwargs=None, label_warp=None,
                       label_shift=None, label_dropout=0,
                       label_crop_rand=None, label_crop_rand2=None,
                       label_crop_left=None, seed=None):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    label_warp (float) : The sigma value of the spline warp applied to
        to the target label mask during training in order to corrupt it. Used
        for testing robustness to label noise.
    label_shift (int) : The number of pixels to shift all training target masks
        to the right.
    label_dropout (float) : The probability in [0, 1] of discarding a slice's
        segmentation mask.
    label_crop_rand (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The minimum size of the rectangle is
        set as a fraction of the connected component's bounding box, in [0, 1].
    label_crop_rand2 (float) : Crop out a randomly sized rectangle out of every
        connected component of the mask. The mean size in each dimension is
        set as a fraction of the connected component's width/height, in [0, 1].
    label_crop_left (float) : If true, crop out the left fraction (in [0, 1]) 
        of every connected component of the mask.
    seed (int) : The seed for the random number generator.
    """
    
    # Set up rng.
    rng = np.random.RandomState(seed)
        
    def process_element(inputs):
        h, s, m, hi, si = inputs
        
        # Drop mask.
        if rng.choice([True, False], p=[label_dropout, 1-label_dropout]):
            m = None
        
        # Crop mask.
        if m is not None and (   label_crop_rand is not None
                              or label_crop_rand2 is not None
                              or label_crop_left is not None):
            m_out = m.copy()
            m_dilated = ndimage.morphology.binary_dilation(m)
            m_labeled, n_obj = ndimage.label(m_dilated)
            for bbox in ndimage.find_objects(m_labeled):
                _, row, col = bbox
                if label_crop_rand is not None:
                    r = int(label_crop_rand*(row.stop-row.start))
                    c = int(label_crop_rand*(col.stop-col.start))
                    row_a = rng.randint(row.start, row.stop+1-r)
                    row_b = rng.randint(row_a+r, row.stop+1)
                    col_a = rng.randint(col.start, col.stop+1-c)
                    col_b = rng.randint(col_a+c, col.stop+1)
                    m_out[:, row_a:row_b, col_a:col_b] = 0
                if label_crop_rand2 is not None:
                    def get_p(n):
                        mu = int(label_crop_rand2*n+0.5)     # mean
                        m = (12*mu-6*n)/(n*(n+1)*(n+2))     # slope
                        i = 1/(n+1)-m*n/2                   # intersection
                        p = np.array([max(x*m+i, 0) for x in range(n+1)])
                        p = p/p.sum()   # Precision errors can make p.sum() > 1
                        return p
                    width  = row.stop - row.start
                    height = col.stop - col.start
                    box_width  = rng.choice(range(width+1),  p=get_p(width))
                    box_height = rng.choice(range(height+1), p=get_p(height))
                    box_row_start = rng.randint(row.start,
                                                row.stop+1-box_width)
                    box_col_start = rng.randint(col.start,
                                                col.stop+1-box_height)
                    row_slice = slice(box_row_start, box_row_start+box_width)
                    col_slice = slice(box_col_start, box_col_start+box_height)
                    m_out[:, row_slice, col_slice] = 0
                if label_crop_left is not None:
                    crop_size = int(label_crop_left*(col.stop-col.start))
                    m_out[:, row, col.start:col.start+crop_size] = 0
            m = m_out
        
        # Float.
        h = h.astype(np.float32)
        s = s.astype(np.float32)
        
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
        
        # Corrupt the mask by warping it.
        if label_warp is not None:
            if m is not None:
                m = image_random_transform(m,
                                           spline_warp=True,
                                           warp_sigma=label_warp,
                                           warp_grid_size=3,
                                           n_warp_threads=1)
        if label_shift is not None:
            if m is not None:
                m_shift = np.zeros(m.shape, dtype=m.dtype)
                m_shift[:,label_shift:,:] = m[:,:-label_shift,:]
                m = m_shift
        
        # Remove distant outlier intensities.
        if h is not None:
            h = np.clip(h, -1., 1.)
        if s is not None:
            s = np.clip(s, -1., 1.)
            
        return h, s, m, hi, si
        
    def process_batch(batch):
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            elem = process_element([b[i] for b in batch])
            elements.append(elem)
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch
