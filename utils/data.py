from __future__ import (print_function,
                        division)
import os
from collections import OrderedDict

import zarr
import imageio
import numpy as np
from skimage import transform

from data_tools.wrap import (delayed_view,
                             multi_source_array)
from data_tools.io import data_flow
from data_tools.data_augmentation import image_stack_random_transform


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


def prepare_data_brats(path_hgg, path_lgg):
    """
    Convenience function to prepare brats data as multi_source_array objects,
    split into training and validation subsets.
    
    path_hgg (string) : Path of the zarr directory containing the HGG data.
    path_lgg (string) : Path of the zarr directory containing the LGG data.
    
    Returns six arrays: healthy slices, sick slices, and segmentations for 
    the training and validation subsets.
    """
    
    # Random 20% data split.
    validation_indices = {'hgg': [60,54,182,64,166,190,184,143,6,75,169,183,
                                  202,166,189,41,158,69,133,180,16,41,0,198,
                                  101,120,5,185,203,151,100,149,44,48,151,34,
                                  88,204,149,119,152,65],
                          'lgg': [25,14,25,4,54,56,56,54,59,1,38,6,24,23,53]}
    orientations = [1,2,3]
    
    
    def _prepare(path, axis, validation_indices):
        # Open zarr file.
        try:
            zgroup = zarr.open_group(path, mode='r')
        except:
            print("Failed to open data: {}".format(path))
            raise
        
        # Assemble volumes and corresponding segmentations.
        volumes_h = []
        volumes_s = []
        segmentations = []
        for key in zgroup.keys():   # Per patient.
            group_p = zgroup[key]
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


def preprocessor_brats(data_augmentation_kwargs=None):
    """
    Preprocessor function to pass to a data_flow, for BRATS data.
    """
    
    def f(batch):
        # Expecting two or three sources in the batch, with np.float32 dtype
        # unless there is a segmentation in which case it is expected to have
        # type np.int64.
        assert len(batch) in [2, 3]
        has_segmentations = False
        if len(batch)==2:
            assert batch[0][0].dtype==np.float32
            assert batch[1][0].dtype==np.float32 or batch[1][0].dtype==np.int64
            if batch[1][0].dtype==np.int64:
                has_segmentations = True
        if len(batch)==3:
            assert batch[0][0].dtype==np.float32
            assert batch[1][1].dtype==np.float32
            assert batch[2][0].dtype==np.float32 or batch[2][0].dtype==np.int64
            if batch[2][0].dtype==np.int64:
                has_segmentations = True
        
        # Center all slices onto empty buffers with a size equal to the
        # largest.
        out_batch = []
        for b in batch:
            # Find the largest slice.
            max_shape = (0,0)
            for im in b:
                im_shape = np.shape(im)[1:]
                max_shape = (max(im_shape[0], max_shape[0]),
                             max(im_shape[1], max_shape[1]))
            
            # Create buffer and center images into it.
            out_b = np.zeros((len(b),len(b[0]))+max_shape, dtype=np.float32)
            out_b[...] = np.inf     # To be replaced with background intensity.
            for i, im in enumerate(b):
                im_shape = np.shape(im)[1:]
                offset = np.subtract(max_shape, im_shape)//2
                index_slices = [slice(None, None),
                                slice(offset[0], offset[0]+im_shape[0]),
                                slice(offset[1], offset[1]+im_shape[1])]
                out_b[i][index_slices] = im[...]
            out_batch.append(out_b)
        
        # Data augmentation.
        if data_augmentation_kwargs is not None:
                
            # Segmentation setup : sick, segmentations
            if len(batch)==2 and has_segmentations:
                out_batch = image_stack_random_transform(out_batch[0],
                                                         y=out_batch[1],
                                                    **data_augmentation_kwargs)
            
            # Translation setup : healthy, sick, (segmentations)
            else:
                b0 = out_batch[0]
                b1 = out_batch[1]
                b2 = None
                if len(out_batch)==3:
                    b2 = out_batch[2]
                    
                b0 = image_stack_random_transform(b0,
                                                  **data_augmentation_kwargs)
                b_ = image_stack_random_transform(b1, y=b2,
                                                  **data_augmentation_kwargs)
                if b2 is None:
                    b1 = b_
                    out_batch = (b0, b1)
                else:
                    b1, b2 = b_
                    out_batch = (b0, b1, b2)
                        
        # Set background intensity in output buffers.
        for b, out_b in zip(batch, out_batch):
            for i, im in enumerate(b):
                # HACK: Assuming corner pixel is always outside of the brain.
                background = im[:,0,0]
                for j in range(len(im)):
                    im_buff = out_b[i,j]
                    im_buff[im_buff==np.inf] = background[j]
                    
        # Remove distant outlier intensities.
        out_batch = list(out_batch)
        out_batch[0] = np.clip(out_batch[0], -2.0, 2.0)
        if len(batch)==3:
            out_batch[1] = np.clip(out_batch[1], -2.0, 2.0)
        if not has_segmentations:
            out_batch[-1] = np.clip(out_batch[-1], -2.0, 2.0)
        
        # Set dtype.
        n = len(batch)
        out_batch = [out_batch[i].astype(batch[i][0].dtype) for i in range(n)]
                        
        return out_batch
    return f
        
