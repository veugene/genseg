from collections import OrderedDict

import h5py
import numpy as np
from scipy import ndimage

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms \
    import (BrightnessMultiplicativeTransform,
            ContrastAugmentationTransform,
            BrightnessTransform)
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms \
    import (GaussianNoiseTransform,
            GaussianBlurTransform)
from batchgenerators.transforms.resample_transforms \
    import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2

from data_tools.data_augmentation import image_random_transform
from data_tools.wrap import multi_source_array


def prepare_data_brats17_no_hemi(path_hgg, path_lgg, masked_fraction=0, drop_masked=False, rng=None):
    # Random 20% data split (10% validation, 10% testing).
    rnd_state = np.random.RandomState(0)
    hgg_indices = np.arange(0, 210)
    lgg_indices = np.arange(0, 75)
    rnd_state.shuffle(hgg_indices)
    rnd_state.shuffle(lgg_indices)
    hgg_val = hgg_indices[0:21]
    lgg_val = lgg_indices[0:7]
    hgg_test = hgg_indices[21:42]
    lgg_test = lgg_indices[7:15]
    validation_indices = {'hgg': hgg_val, 'lgg': lgg_val}
    testing_indices = {'hgg': hgg_test, 'lgg': lgg_test}
    return _prepare_data_brats(path_hgg, path_lgg,
                               masked_fraction=masked_fraction,
                               validation_indices=validation_indices,
                               testing_indices=testing_indices,
                               drop_masked=drop_masked,
                               rng=rng)


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
        assert len_s > 0
        if len_h == 0:
            # Create a dummy volume.
            print('Warning: no healthy volumes found! Creating a dummy empty '
                  'volume.')
            volumes_h[key] = [np.zeros((1,)+volumes_s[key][0].shape[1:],
                                       dtype=volumes_s[key][0].dtype)]
            indices_h[key] = [np.zeros(1, dtype=np.uint32)]
            len_h = 1
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
                       label_crop_left=None):
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
    """
        
    def process_element(inputs):
        h, s, m, hi, si = inputs
        
        # Set up rng.
        if m is not None:
            seed = abs(hash(m.data.tobytes()))//2**32
            rng = np.random.RandomState(seed)
        
        # Drop mask.
        if m is not None:
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
        if data_augmentation_kwargs=='nnunet':
            if h is not None:
                h = nnunet_transform(h)
            if s is not None:
                _ = nnunet_transform(s, m)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
        elif data_augmentation_kwargs=='nnunet_default':
            if h is not None:
                h = nnunet_transform_default(h)
            if s is not None:
                _ = nnunet_transform_default(s, m)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
        elif data_augmentation_kwargs=='nnunet_default_3d':
            if h is not None:
                h = nnunet_transform_default_3d(h)
            if s is not None:
                _ = nnunet_transform_default_3d(s, m)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
        elif data_augmentation_kwargs is not None:
            if h is not None:
                h = image_random_transform_(h, **data_augmentation_kwargs,
                                            n_warp_threads=1)
            if s is not None:
                _ = image_random_transform_(s, m, **data_augmentation_kwargs,
                                            n_warp_threads=1)
            if m is not None:
                assert s is not None
                s, m = _
            else:
                s = _
        
        # Corrupt the mask by warping it.
        if label_warp is not None:
            if m is not None:
                m = image_random_transform_(m,
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


def nnunet_transform(img, seg=None):
    # Based on `data_augmentation_insaneDA2.py` and on "data_aug_params"
    # extracted from the pretrained BRATS nnunet:
    #
    #"data_aug_params":
    #"{
    #'selected_data_channels': None,
    #'selected_seg_channels': [0],
    #'do_elastic': True,
    #'elastic_deform_alpha': (0.0, 900.0),
    #'elastic_deform_sigma': (9.0, 13.0),
    #'p_eldef': 0.3,
    #'do_scaling': True, 
    #'scale_range': (0.65, 1.6),
    #'independent_scale_factor_for_each_axis': True,
    #'p_independent_scale_per_axis': 0.3,
    #'p_scale': 0.3,
    #'do_rotation': True,
    #'rotation_x': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_y': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_z': (-0.5235987755982988, 0.5235987755982988),
    #'rotation_p_per_axis': 1,
    #'p_rot': 0.3,
    #'random_crop': False,
    #'random_crop_dist_to_border': None,
    #'do_gamma': True,
    #'gamma_retain_stats': True,
    #'gamma_range': (0.5, 1.6),
    #'p_gamma': 0.3,
    #'do_mirror': True,
    #'mirror_axes': (0, 1, 2),
    #'dummy_2D': False,
    #'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
    #'border_mode_data': 'constant',
    #'all_segmentation_labels': None,
    #'move_last_seg_chanel_to_data': False,
    #'cascade_do_cascade_augmentations': False,
    #'cascade_random_binary_transform_p': 0.4,
    #'cascade_random_binary_transform_p_per_label': 1,
    #'cascade_random_binary_transform_size': (1, 8),
    #'cascade_remove_conn_comp_p': 0.2,
    #'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
    #'cascade_remove_conn_comp_fill_with_other_class_p': 0.0,
    #'do_additive_brightness': True,
    #'additive_brightness_p_per_sample': 0.3,
    #'additive_brightness_p_per_channel': 1,
    #'additive_brightness_mu': 0,
    #'additive_brightness_sigma': 0.2,
    #'num_threads': 24,
    #'num_cached_per_thread': 4,
    #'patch_size_for_spatialtransform': array([128, 128, 128]),
    #'eldef_deformation_scale': (0, 0.25)
    #}",
    #
    #
    # NOTE: scale has been reduced from (0.65, 1.6) to (0.9, 1.1) in order
    # to make sure that tumour is never removed from sick images.
    # 
    transforms = []
    transforms += [
        SpatialTransform_2(
            patch_size=None,
            do_elastic_deform=True,
            deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(-30/360*2*np.pi, 30/360*2*np.pi),
            angle_y=(-30/360*2*np.pi, 30/360*2*np.pi),
            angle_z=(-30/360*2*np.pi, 30/360*2*np.pi),
            do_scale=True,
            #scale=(0.65, 1.6),
            scale=(0.9, 1.1),
            border_mode_data='constant',
            border_cval_data=0,
            order_data=3,
            border_mode_seg='constant',
            border_cval_seg=0,
            order_seg=0,
            random_crop=False,
            p_el_per_sample=0.3,
            p_scale_per_sample=0.3,
            p_rot_per_sample=0.3,
            independent_scale_for_each_axis=False,
            p_independent_scale_per_axis=0.3
        )
    ]
    transforms += [GaussianNoiseTransform(p_per_sample=0.15)]
    transforms += [
        GaussianBlurTransform(
            (0.5, 1.5),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5
        )
    ]
    transforms += [
        BrightnessMultiplicativeTransform(
            multiplier_range=(0.70, 1.3),
            p_per_sample=0.15
        )
    ]
    transforms += [
        ContrastAugmentationTransform(
            contrast_range=(0.65, 1.5),
            p_per_sample=0.15
        )
    ]
    transforms += [
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
            ignore_axes=None
        )
    ]
    transforms += [
        GammaTransform(     # This one really does appear twice...
            (0.5, 1.6),     # gamma_range
            True,
            True,
            retain_stats=True,
            p_per_sample=0.15   # Hardcoded.
        )
    ]
    transforms += [
        BrightnessTransform(
            0,      # additive_brightness_mu
            0.2,    # additive_brightness_sigma
            True,
            p_per_sample=0.3,
            p_per_channel=1
        )
    ]
    transforms += [
        GammaTransform(
            (0.5, 1.6),     # gamma_range
            False,
            True,
            retain_stats=True,
            p_per_sample=0.3    # Passed as param.
        )
    ]
    transforms += [MirrorTransform((1, 2))]  # mirror_axes
    full_transform = Compose(transforms)
    
    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output

def nnunet_transform_default(img, seg=None):
    #
    #"data_aug_params": "{'selected_data_channels': None,
    # 'selected_seg_channels': [0],
    # 'do_elastic': False,
    # 'elastic_deform_alpha': (0.0, 200.0),
    # 'elastic_deform_sigma': (9.0, 13.0),
    # 'do_scaling': True,
    # 'scale_range': (0.7, 1.4),
    # 'do_rotation': True,
    # 'rotation_x': (-3.141592653589793, 3.141592653589793),
    # 'rotation_y': (-0.0, 0.0),
    # 'rotation_z': (-0.0, 0.0),
    # 'random_crop': False,
    # 'random_crop_dist_to_border':  # None,
    # 'do_gamma': True,
    # 'gamma_retain_stats': True,
    # 'gamma_range': (0.7, 1.5),
    # 'p_gamma': 0.3,
    # 'num_threads': 12,
    # 'num_cached_per_thread': 1,
    # 'do_mirror': True,
    # 'mirror_axes': (0, 1),
    # 'p_eldef': 0.2,
    # 'p_scale': 0.2,
    # 'p_rot': 0.2,
    # 'dummy_2D': False,
    # 'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
    # 'all_segmentation_labels': None,
    # 'move_last_seg_chanel_to_data': False,
    # 'border_mode_data': 'constant',
    # 'cascade_do_cascade_augmentations': False,
    # 'cascade_random_binary_transform_p': 0.4,
    # 'cascade_random_binary_transform_size': (1, 8),
    # 'cascade_remove_conn_comp_p': 0.2,
    # 'cascade_remove_conn_comp_max_size_percent_threshold': 0.15,
    # 'cascade_remove_conn_comp_fill_with_other_class_p': 0.0,
    # 'independent_scale_factor_for_each_axis': False,
    # 'patch_size_for_spatialtransform': array([192, 160])}",

    transforms = []
    transforms += [SpatialTransform(
        None,
        patch_center_dist_from_border=None,
        do_elastic_deform=False,
        alpha=(0.0, 200.0),
        sigma=(9.0, 13.0),
        do_rotation=True, angle_x=(-0.2617993877991494, 0.2617993877991494), angle_y=(-0.0, 0.0),
        angle_z=(-0.0, 0.0), p_rot_per_axis=1,
        do_scale=True, scale=(0.9, 1.1),
        border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg="constant", border_cval_seg=0,
        order_seg=0, random_crop=False, p_el_per_sample=0.2,
        p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=True
    )]
    transforms += [
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                              p_per_channel=0.5),
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
    ]
    transforms += [(ContrastAugmentationTransform(p_per_sample=0.15)),
                   SimulateLowResolutionTransform(zoom_range=(0.5, 1),
                                                  per_channel=True, p_per_channel=0.5,
                                                  order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                  ignore_axes=None)]
    transforms += [
        GammaTransform((0.7, 1.5),
                       True,
                       True,
                       retain_stats=True,
                       p_per_sample=0.1)
    ]

    transforms += [
        GammaTransform((0.7, 1.5),
                       False,
                       True,
                       retain_stats=True,
                       p_per_sample=0.3)
    ]

    transforms += [
        MirrorTransform((0,1))
    ]

    # transforms += [
    #     MaskTransform(OrderedDict([(0, False)]), mask_idx_in_seg=0, set_outside_to=0)
    # ]

    # TODO; check whether delete or not
    # transforms += [
    #    RemoveLabelTransform(-1, 0)
    # ]
    # transforms += [
    #    RenameTransform('seg', 'target', True)
    # ]

    full_transform = Compose(transforms)

    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output

def nnunet_transform_default_3d(img, seg=None, border_val_seg=-1, order_seg=0, order_data=3):
    params = {'selected_data_channels': None, 'selected_seg_channels': [0], 'do_elastic': False,
              'elastic_deform_alpha': (0.0, 900.0), 'elastic_deform_sigma': (9.0, 13.0), 'do_scaling': True,
              'scale_range': (0.7, 1.4), 'do_rotation': True, 'rotation_x': (-0.5235987755982988, 0.5235987755982988),
              'rotation_y': (-0.5235987755982988, 0.5235987755982988),
              'rotation_z': (-0.5235987755982988, 0.5235987755982988), 'random_crop': False,
              'random_crop_dist_to_border': None, 'do_gamma': True, 'gamma_retain_stats': True,
              'gamma_range': (0.7, 1.5), 'p_gamma': 0.3, 'num_threads': 12, 'num_cached_per_thread': 1,
              'do_mirror': True, 'mirror_axes': (0, 1, 2), 'p_eldef': 0.2, 'p_scale': 0.2, 'p_rot': 0.2,
              'dummy_2D': False,
              'mask_was_used_for_normalization': OrderedDict([(0, True), (1, True), (2, True), (3, True)]),
              'all_segmentation_labels': None, 'move_last_seg_chanel_to_data': False, 'border_mode_data': 'constant',
              'cascade_do_cascade_augmentations': False,
              'patch_size_for_spatialtransform': np.array([128, 128, 128])}
    ignore_axes = None

    transforms = []

    transforms += [
        SpatialTransform(
            None, patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"),
            do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg,
            order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=True, p_rot_per_axis=1)
    ]

    transforms += [
        GaussianNoiseTransform(p_per_sample=0.1)
    ]

    transforms += [
        GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                              p_per_channel=0.5)
    ]

    transforms += [
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
    ]

    if params.get("do_additive_brightness"):
        transforms += [
            BrightnessTransform(params.get("additive_brightness_mu"),
                                params.get("additive_brightness_sigma"),
                                True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                p_per_channel=params.get("additive_brightness_p_per_channel"))
        ]

    transforms += [ContrastAugmentationTransform(p_per_sample=0.15)]
    transforms += [
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                       p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=ignore_axes)
    ]

    transforms += [
        GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                       p_per_sample=0.1)
    ]

    if params.get("do_gamma"):
       transforms += [
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"])
       ]

    if params.get("do_mirror") or params.get("mirror"):
        transforms += [
            MirrorTransform((0,1))
        ]

    #if params.get("mask_was_used_for_normalization") is not None:
    #    mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
    #    transforms += [
    #        MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0)
    #    ]

    full_transform = Compose(transforms)

    # Transform.
    img_input = img[None]
    seg_input = seg
    if seg is not None:
        seg_input = seg[None]
    out = full_transform(data=img_input, seg=seg_input)
    img_output = out['data'][0]
    if seg is None:
        return img_output
    seg_output = out['seg'][0]
    return img_output, seg_output


def image_random_transform_(x, y=None, **kwargs):
    x_shape = x.shape
    if x.ndim > 3:
        x = np.reshape(x, (np.prod(x_shape[:-2]), x_shape[-2], x_shape[-1]))
    if y is not None:
        y_shape = y.shape
        if y.ndim > 3:
            y = np.reshape(y, (np.prod(y_shape[:-2]), y_shape[-2], y_shape[-1]))
    if y is not None:
        x, y = image_random_transform(x, y, **kwargs)
    else:
        x = image_random_transform(x, **kwargs)
    if x.shape != x_shape:
        x = np.reshape(x, x_shape)
    if y is not None and y.shape != y_shape:
            y = np.reshape(y, y_shape)
    if y is not None:
        return x, y
    return x
    