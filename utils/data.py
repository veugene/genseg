from __future__ import (print_function,
                        division)
import os
from collections import OrderedDict

import h5py
import imageio
import numpy as np
from skimage import transform

from data_tools.wrap import (delayed_view,
                             multi_source_array)
from data_tools.io import data_flow
from data_tools.data_augmentation import image_random_transform


def resize_stack(arr, size, interp='bilinear'):
    """
    Resize each slice of a tensor, assuming the last two dimensions span the
    slice.
    """
    out = np.zeros(arr.shape[:-2]+size, dtype=arr.dtype)
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = transform.resize(arr[idx],
                                    output_shape=size,
                                    mode='constant',
                                    cval=0,
                                    clip=True,
                                    preserve_range=True)
    return out


class folder_indexer(object):
    """
    Single-value index interface for loading images from a folder.
    """
    def __init__(self, path):
        self.path = path
        self.convert_rgb = convert_rgb
        self.fn_list = sorted(os.listdir(path))
        self.shape = (len(self.fn_list),)
        self.dtype = 'strings'
      
    def __getitem__(self, index):
        fn = self.fn_list[index]
        return imageio.imread(os.path.join(path, img_fn))
    
    def __len__(self):
        return len(self.fn_list)


class data_flow_sampler(data_flow):
    """
    A data_flow implementation that does not restrict every source in `data`
    to be of equal length. 
    
    Whenever an iterator is created by data_flow (eg. each epoch), a random
    (equal length) non-contiguous subset of data is viewed from each source
    that is longer than the shortest length.
    
    NOTE: this means that these sources are shuffled, regardless of which
          sampling arguments are passed.
    """
    def __init__(self, data, *args, **kwargs):
        min_length = min(len(d) for d in data)
        _data = []
        self.reshuffle_indicator = []
        for d in data:
            if len(d) > min_length:
                d = delayed_view(d, shuffle=True)
                d = delayed_view(d, idx_min=0, idx_max=min_length)
                self.reshuffle_indicator.append(True)
            else:
                self.reshuffle_indicator.append(False)
            _data.append(d)
            
        super(data_flow_sampler, self).__init__(data=_data, *args, **kwargs)
        
    def flow(self):
        for i, reshuffle in enumerate(self.reshuffle_indicator):
            if reshuffle:
                self.data[i].arr.re_shuffle()
        return super(data_flow_sampler, self).flow()


def prepare_data_brats(path_hgg, path_lgg, orientations=None):
    """
    Convenience function to prepare brats data as multi_source_array objects,
    split into training and validation subsets.
    
    path_hgg (string) : Path of the h5py file containing the HGG data.
    path_lgg (string) : Path of the h5py file containing the LGG data.
    orientations (list) : A list of integers in {1, 2, 3}, specifying the
        axes along which to slice image volumes.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    if orientations is None:
        orientations = [1,2,3]
    
    # Random 20% data split.
    validation_indices = {'hgg': [60,54,182,64,166,190,184,143,6,75,169,183,
                                  202,166,189,41,158,69,133,180,16,41,0,198,
                                  101,120,5,185,203,151,100,149,44,48,151,34,
                                  88,204,149,119,152,65],
                          'lgg': [25,14,25,4,54,56,56,54,59,1,38,6,24,23,53]}
    
    
    def _prepare(path, axis, validation_indices):
        # Open h5py file.
        try:
            h5py_file = h5py.File(path, mode='r')
        except:
            print("Failed to open data: {}".format(path))
            raise
        
        # Assemble volumes and corresponding segmentations.
        volumes_h = []
        volumes_s = []
        segmentations = []
        for key in h5py_file.keys():   # Per patient.
            group_p = h5py_file[key]
            volumes_h.append(group_p['healthy/axis_{}'.format(str(axis))])
            volumes_s.append(group_p['sick/axis_{}'.format(str(axis))])
            segmentations.append(\
                        group_p['segmentation/axis_{}'.format(str(axis))])
        
        # Split data into training and validation.
        training_indices = [i for i in range(len(segmentations)) \
                            if i not in validation_indices]
        h_train = [volumes_h[i] for i in training_indices]
        h_valid = [volumes_h[i] for i in validation_indices]
        s_train = [volumes_s[i] for i in training_indices]
        s_valid = [volumes_s[i] for i in validation_indices]
        m_train = [segmentations[i] for i in training_indices]
        m_valid = [segmentations[i] for i in validation_indices]
        
        return h_train, h_valid, s_train, s_valid, m_train, m_valid
    
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
        _extend(data_hgg, _prepare(path_hgg, axis, validation_indices['hgg']))
        _extend(data_lgg, _prepare(path_lgg, axis, validation_indices['lgg']))
    
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict())])
    msa = multi_source_array
    data['train']['h'] = msa(data_hgg[0]+data_lgg[0], no_shape=True)
    data['valid']['h'] = msa(data_hgg[1]+data_lgg[1], no_shape=True)
    data['train']['s'] = msa(data_hgg[2]+data_lgg[2], no_shape=True)
    data['valid']['s'] = msa(data_hgg[3]+data_lgg[3], no_shape=True)
    data['train']['m'] = msa(data_hgg[4]+data_lgg[4], no_shape=True)
    data['valid']['m'] = msa(data_hgg[5]+data_lgg[5], no_shape=True)
        
    return data


def preprocessor_brats(data_augmentation_kwargs=None,
                       h_idx=0, s_idx=1, m_idx=2):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    
    data_augmentation_kwargs : Dictionary of keyword arguments to pass to
        the data augmentation code (image_stack_random_transform).
    h_idx : The batch index for the healthy slices (None if not in batch).
    s_idx : The batch index for the sick slices (None if not in batch).
    m_idx : The batch index for the segmentation masks (None if not in batch).
    """
        
    def process_element(h, s, m, max_shape):
        inputs = [h, s, m]
        
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
            elem_im[index_slices] = im[...]
            elem.append(elem_im)
        h, s, m = elem
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
            if h is not None:
                h = image_random_transform(h, **data_augmentation_kwargs)
            if s is not None:
                _ = image_random_transform(s, m, **data_augmentation_kwargs)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
                        
        # Set background intensity.
        for im, im_orig in zip([h, s, m], inputs):
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
            im = np.clip(im, -2., 2.)
            hs.append(im)
        h, s = hs
        
        # Set dtype (all output buffers are float32 to support inf).
        elem = []
        for im, im_orig in zip([h, s, m], inputs):
            if im is None:
                elem.append(None)
                continue
            im = im.astype(im_orig.dtype)
            elem.append(im)
            
        return elem
        
    def process_batch(batch):        
        # Find the largest slice.
        max_shape = (0,0)
        for b in batch:
            for im in b:
                if im is None:
                    continue
                im_shape = np.shape(im)[1:]
                max_shape = (max(im_shape[0], max_shape[0]),
                             max(im_shape[1], max_shape[1]))
                
        # Process every element.
        elements = []
        for i in range(len(batch[0])):
            h = None if h_idx is None else batch[h_idx][i]
            s = None if s_idx is None else batch[s_idx][i]
            m = None if m_idx is None else batch[m_idx][i]
            elem = process_element(h=h, s=s, m=m, max_shape=max_shape)
            elements.append(elem)
            
        out_batch = list(zip(*elements))
        return out_batch
    
    return process_batch
        

class masked_view(delayed_view):
    """
    Given an array, mask some random subset of indices, returning `None`
    for at these indices.
    
    arr : The source array.
    masked_fraction : Fraction of elements to mask (length axis), in [0, 1.0].
    rng : A numpy random number generator.
    """
    
    def __init__(self, arr, masked_fraction, rng=None):
        super(masked_view, self).__init__(arr=arr, shuffle=False, rng=rng)
        if 0 > masked_fraction or masked_fraction > 1:
            raise ValueError("In `masked_view`, `masked_fraction` must be set "
                             "to a value in [0, 1.0] but was set to {}."
                             "".format(masked_fraction))
        self.masked_fraction = masked_fraction
        num_masked_indices = int(min(self.num_items,
                                     masked_fraction*self.num_items+0.5))
        self.masked_indices = self.rng.choice(self.num_items,
                                              size=num_masked_indices,
                                              replace=False)
        
    def _get_element(self, int_key, key_remainder=None):
        if not isinstance(int_key, (int, np.integer)):
            raise IndexError("cannot index with {}".format(type(int_key)))
        if int_key in self.masked_indices:
            return None
        idx = self.arr_indices[int_key]
        if key_remainder is not None:
            idx = (idx,)+key_remainder
        idx = int(idx)  # Some libraries don't like np.integer
        return self.arr[idx]
