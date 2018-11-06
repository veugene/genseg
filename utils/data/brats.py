from collections import OrderedDict

import h5py
import numpy as np

from data_tools.wrap import multi_source_array
from data_tools.data_augmentation import image_random_transform

 
def prepare_data_brats17(path_hgg, path_lgg,
                         masked_fraction=0, drop_masked=False,
                         orientations=None,
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
                               orientations=orientations,
                               rng=rng)

def prepare_data_brats13s(path_hgg, path_lgg,
                          masked_fraction=0, drop_masked=False,
                          orientations=None,
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
                               orientations=orientations,
                               rng=rng)

def _prepare_data_brats(path_hgg, path_lgg, validation_indices,
                        masked_fraction=0, drop_masked=False,
                        orientations=None, rng=None):
    """
    Convenience function to prepare brats data as multi_source_array objects,
    split into training and validation subsets.
    
    path_hgg (string) : Path of the h5py file containing the HGG data.
    path_lgg (string) : Path of the h5py file containing the LGG data.
    masked_fraction (float) : The fraction in [0, 1.] of volumes in the 
        training set for which  to return segmentation masks as None
    drop_masked (bool) : Whether to omit volumes with "masked" segmentations.
    orientations (list) : A list of integers in {1, 2, 3}, specifying the
        axes along which to slice image volumes.
    rng (numpy RandomState) : Random number generator.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    if orientations is None:
        orientations = [1,2,3]
    elif not hasattr(orientations, '__len__'):
        orientations = [orientations]
    if rng is None:
        rng = np.random.RandomState()
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
    
    training_indices = {'hgg': [i for i in range(num_hgg) \
                                if i not in validation_indices['hgg']],
                        'lgg': [i for i in range(num_lgg) \
                                if i not in validation_indices['lgg']]}
    
    # Volumes with these indices will either be dropped from the training_set
    # or have their segmentations set to None.
    masked_indices = {
        'hgg': rng.choice(num_hgg,
                          size=int(min(num_hgg, masked_fraction*num_hgg+0.5)),
                          replace=False),
        'lgg': rng.choice(num_lgg,
                          size=int(min(num_lgg, masked_fraction*num_lgg+0.5)),
                          replace=False)
        }
    if drop_masked:
        for key in masked_indices.keys():
            training_indices[key] = [i for i in training_indices[key] \
                                     if i not in masked_indices[key]]
            masked_indices[key] = []
    
    def _prepare(path, axis, key):
        # Open h5py file.
        try:
            h5py_file = h5py.File(path, mode='r')
        except:
            print("Failed to open data: {}".format(path))
            raise
        
        # Assemble volumes and corresponding segmentations.
        volumes_h = []
        volumes_s = []
        volumes_m = []
        indices_h = []
        indices_s = []
        for _key in h5py_file.keys():   # Per patient.
            group_p = h5py_file[_key]
            volumes_h.append(group_p['healthy/axis_{}'.format(str(axis))])
            volumes_s.append(group_p['sick/axis_{}'.format(str(axis))])
            volumes_m.append(group_p['segmentation/axis_{}'.format(str(axis))])
            indices_h.append(group_p['h_indices/axis_{}'.format(str(axis))])
            indices_s.append(group_p['s_indices/axis_{}'.format(str(axis))])
        
        # Split data into training and validation.
        h_train = [volumes_h[i] for i in training_indices[key]]
        h_valid = [volumes_h[i] for i in validation_indices[key]]
        s_train = [volumes_s[i] for i in training_indices[key]]
        s_valid = [volumes_s[i] for i in validation_indices[key]]
        m_train = [volumes_m[i] if i not in masked_indices[key]
                   else np.array([None]*len(volumes_m[i]))
                   for i in training_indices[key]]
        m_valid = [volumes_m[i] for i in validation_indices[key]]
        hi_train = [indices_h[i] for i in training_indices[key]]
        hi_valid = [indices_h[i] for i in validation_indices[key]]
        si_train = [indices_s[i] for i in training_indices[key]]
        si_valid = [indices_s[i] for i in validation_indices[key]]
        
        return (h_train,  h_valid,
                s_train,  s_valid,
                m_train,  m_valid,
                hi_train, hi_valid,
                si_train, si_valid)
    
    def _extend(target, source):
        for i in range(len(source)):
            if len(target) < len(source):
                target.append(source[i])
            else:
                target[i].extend(source[i])
        return target
    
    data_hgg = []
    data_lgg = []
    for axis in orientations:
        _extend(data_hgg, _prepare(path_hgg, axis=axis, key='hgg'))
        _extend(data_lgg, _prepare(path_lgg, axis=axis, key='lgg'))
    
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict())])
    msa = multi_source_array
    train_h  = data_hgg[0]+data_lgg[0]
    valid_h  = data_hgg[1]+data_lgg[1]
    train_s  = data_hgg[2]+data_lgg[2]
    valid_s  = data_hgg[3]+data_lgg[3]
    train_m  = data_hgg[4]+data_lgg[4]
    valid_m  = data_hgg[5]+data_lgg[5]
    train_hi = data_hgg[6]+data_lgg[6]
    valid_hi = data_hgg[7]+data_lgg[7]
    train_si = data_hgg[8]+data_lgg[8]
    valid_si = data_hgg[9]+data_lgg[9]

    # HACK: we may have a situation where the number of sick examples
    # is greater than the number of healthy. In that case, we should
    # duplicate the healthy set M times so that it has a bigger size
    # than the sick set.
    m = 1
    len_h = sum([ len(elem) for elem in train_h])
    len_s = sum([ len(elem) for elem in train_s])
    if len_h < len_s:
        m = int(np.ceil(len_s / len_h))
    data['train']['h']  = msa(train_h*m, no_shape=True)
    data['valid']['h']  = msa(valid_h*m, no_shape=True)
    data['train']['s']  = msa(train_s, no_shape=True)
    data['valid']['s']  = msa(valid_s, no_shape=True)
    data['train']['m']  = msa(train_m, no_shape=True)
    data['valid']['m']  = msa(valid_m, no_shape=True)
    data['train']['hi'] = msa(train_hi, no_shape=True)
    data['valid']['hi'] = msa(valid_hi, no_shape=True)
    data['train']['si'] = msa(train_si, no_shape=True)
    data['valid']['si'] = msa(valid_si, no_shape=True)
        
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
