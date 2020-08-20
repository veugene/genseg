import argparse
from collections import (defaultdict,
                         OrderedDict)
import os
import shutil
import re
import tempfile

import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from tqdm import tqdm

from data_tools.io import h5py_array_writer
from data_tools.wrap import multi_source_array
from ddsm_convert.make_dataset import make_data_set as convert_raws


def get_parser():
    parser = argparse.ArgumentParser(description=""
        "Create an HDF5 dataset with DDSM data by using all healthy images "
        "in addition to the sick images that exist also in CBIS-DDSM. The "
        "sick images are sourced from the original DDSM data to ensure that "
        "they are processed in the same way as healthy images but their "
        "annotations are sourced from CBIS-DDSM. All images are cropped to "
        "remove irrelevant background and resized to a square.")
    parser.add_argument('path_raws', type=str,
                        help="path to the DDSM raw data directory")
    parser.add_argument('path_cbis', type=str,
                        help="path to the CBIS-DDSM data directory")
    parser.add_argument('--path_processed', type=str,
                        help="path to save processed raw DDSM data to"
                             "(if not specified, will create a temporary "
                             "directory and delete it when done)")
    parser.add_argument('--path_create', type=str,
                        default='./data/ddsm/ddsm.h5',
                        help="path to save the HDF5 file to")
    parser.add_argument('--force', action='store_true',
                        help="force the overwrite of any existing files "
                             "in `--path_processed`")
    parser.add_argument('--validation_fraction', type=float, default=0.2,
                        help="fraction of (sick, CBIS-DDSM) training data "
                             "to use for the validation subset")
    parser.add_argument('--clip_noise', type=bool, default=True,
                        help="clip out low and high values of image after "
                             "conversion from raw for 'noise reduction'")
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser


def prepare_data_ddsm(args, path_processed):
    
    # Create HDF5 file to save dataset into.
    dir_create = os.path.dirname(args.path_create)
    if not os.path.exists(dir_create):
        os.makedirs(dir_create)
    f = h5py.File(args.path_create, 'w')
    f.create_group('train')
    f.create_group('valid')
    f.create_group('test')
    
    # Create batch-wise writers for the HDF5 file.
    writer = {'train': {}, 'valid': {}, 'test': {}}
    writer_kwargs = {'data_element_shape': (args.resize, args.resize),
                     'batch_size': args.batch_size,
                     'filename': args.path_create,
                     'append': True,
                     'kwargs': {'chunks': (args.batch_size,
                                           args.resize,
                                           args.resize)}}
    writer['train']['h'] = h5py_array_writer(array_name='train/h',
                                             dtype=np.uint16,
                                             **writer_kwargs)
    for fold in ['train', 'valid', 'test']:
        for key in ['s', 'm']:
            writer[fold][key] = h5py_array_writer(
                array_name='{}/{}'.format(fold, key),
                dtype=np.uint16 if key=='s' else np.uint8,
                **writer_kwargs)
    
    # Index all processed DDSM files.
    files_ddsm_s = {}
    files_ddsm_h = []
    for root, dirs, files in os.walk(os.path.join(path_processed)):
        for fn in files:
            if not fn.endswith(".tif"):
                # Not a processed image; skip.
                continue
            if root.startswith(os.path.join(path_processed, "normals")):
                # Healthy case; store in a list.
                files_ddsm_h.append(os.path.join(root, fn))
                continue
            
            # Sick case; index.
            fn_base = os.path.basename(fn)
            case = re.search("(?<=^[A-Z]_)\d{4}", fn_base).group(0)
            view = re.search("(LEFT_CC|LEFT_MLO|RIGHT_CC|RIGHT_MLO)",
                             fn_base).group(0)
            files_ddsm_s["{}_{}".format(case, view)] = os.path.join(root, fn)
            #print("DEBUG {}_{}".format(case, view))
    
    # Collect and store healthy cases (resized).
    print("Processing normals (healthy) from DDSM")
    for im_path in tqdm(files_ddsm_h):
        im = sitk.ReadImage(im_path)
        im_np = sitk.GetArrayFromImage(im)
        im_np = trim(im_np)
        im_np = resize(im_np,
                       size=(args.resize, args.resize),
                       interpolator=sitk.sitkLinear)
        writer['train']['h'].buffered_write(im_np)
    writer['train']['h'].flush_buffer()
    
    # Collect sick cases.
    dirs_image = {'train': [], 'test': []}
    dirs_mask  = {'train': defaultdict(list), 'test': defaultdict(list)}
    keys = {'train': [], 'test': []}
    for d in sorted(os.listdir(args.path_cbis)):
        if not os.path.isdir(os.path.join(args.path_cbis, d)):
            continue
        if re.search("^Mass-(Training|Test).*[1-9]$", d):
            key = re.search("\d{4}_(LEFT_CC|LEFT_MLO|RIGHT_CC|RIGHT_MLO)",
                            d).group(0)
            if re.search("Training", d):
                dirs_mask['train'][key].append(d)
                keys['train'].append(key)
            else:
                dirs_mask['test'][key].append(d)
                keys['test'].append(key)
    
    # Split sick training cases into training and validation subsets.
    keys_train = sorted(keys['train'])
    rng = np.random.RandomState(0)
    rng.shuffle(keys_train)
    n_valid = int(args.validation_fraction*len(keys_train))
    keys_valid, keys_train = keys_train[:n_valid], keys_train[n_valid:]
    keys = {'train': keys_train,
            'valid': keys_valid,
            'test' : keys['test']}
    
    # Create new sick cases dicts with validation subset.
    dirs_mask_split = {
        'train': dict([(key, dirs_mask['train'][key]) for key in keys_train]),
        'valid': dict([(key, dirs_mask['train'][key]) for key in keys_valid]),
        'test': dirs_mask['test']
        }
    
    # For each case, combine all lesions into one mask; process image 
    # and mask and store in HDF5 file.
    for fold in ['train', 'valid', 'test']:
        print("Processing cases with masses (sick) from CBIS-DDSM : '{}' fold"
              "".format(fold))
        fail_list = []
        for key in tqdm(sorted(keys[fold])):
            mask_dir_list = dirs_mask_split[fold][key]
            
            # Function to get path for every dcm image in a directory tree.
            def get_all_dcm(path):
                dcm_list = []
                for root, dirs, files in os.walk(path):
                    for fn in files:
                        if fn.endswith('.dcm'):
                            dcm_list.append(os.path.join(root, fn))
                return dcm_list
            
            # Load image, resize, and store in HDF5.
            try:
                im_path = files_ddsm_s[key]
            except KeyError:
                # The raw DDSM data is missing this case from CBIS-DDSM.
                # (Whyyyy???)
                fail_list.append(key)
            im = sitk.ReadImage(im_path)
            im_np = sitk.GetArrayFromImage(im)
            
            # Load all masks, merge them together, resize, and store in HDF5.
            mask_np = None
            for d in mask_dir_list:
                m_np = None
                mask_path = get_all_dcm(os.path.join(args.path_cbis, d))
                assert len(mask_path)==2  # there should be two dcm files
                m0 = sitk.ReadImage(mask_path[0])
                m0_np = sitk.GetArrayFromImage(m0)
                m1 = sitk.ReadImage(mask_path[0])
                m1_np = sitk.GetArrayFromImage(m1)
                # Of the two files, the mask is the one with the same
                # image size as the full xray image.
                assert not (    m0_np.shape==im_np.shape
                            and m1_np.shape==im_np.shape)
                if m0_np.shape==im_np.shape:
                    m_np = m0_np
                if m1_np.shape==im_np.shape:
                    m_np = m1_np
                if m_np is None:
                    # Could not find a mask matching the image dimensions!
                    # Don't bother looking at the other masks. Just fail this
                    # case. And hope the other cases have correct masks.
                    break
                m_np = np.squeeze(m_np)
                if mask_np is None:
                    mask_np = m_np
                else:
                    mask_np = np.logical_or(m_np, mask_np).astype(np.uint8)
            if mask_np is None:
                # One or more masks does not match the image dimensions.
                # Skip this case and add it to the fail list. :(
                fail_list.append(key)
                continue
            im_np, mask_np = trim(im_np, mask_np)
            im_np = resize(im_np,
                           size=(args.resize, args.resize, 1),
                           interpolator=sitk.sitkLinear)
            im_np = np.squeeze(im_np)
            mask_np = resize(mask_np,
                             size=(args.resize, args.resize),
                             interpolator=sitk.sitkNearestNeighbor)
            writer[fold]['s'].buffered_write(im_np)
            writer[fold]['m'].buffered_write(mask_np)
        writer[fold]['s'].flush_buffer()
        writer[fold]['m'].flush_buffer()
    
        # Report failed cases.
        if len(fail_list):
            print("Skipped {} cases due to issues with their masks (missing "
                  "mask or incorrect size):\n{}"
                  "".format(len(fail_list),
                            "\n".join(fail_list)))


def resize(image, size, interpolator=sitk.sitkLinear):
    sitk_image = image
    if isinstance(image, np.ndarray):
        sitk_image = sitk.GetImageFromArray(image)
    new_spacing = [x*y/z for x, y, z in zip(
                   sitk_image.GetSpacing(),
                   sitk_image.GetSize(),
                   size)]
    sitk_out = sitk.Resample(sitk_image,
                             size,
                             sitk.Transform(),
                             interpolator,
                             sitk_image.GetOrigin(),
                             new_spacing,
                             sitk_image.GetDirection(),
                             0,
                             sitk_image.GetPixelID())
    if isinstance(image, np.ndarray):
        out = sitk.GetArrayFromImage(sitk_out)
        return out
    return sitk_out


def trim(image, mask=None):
    if mask is not None:
        assert np.all(mask.shape==image.shape)
    
    # Normalize.
    x = image.copy()
    if x.dtype==np.uint8:
        x = (x-63)/(2**8 - 1)
    elif x.dtype==np.uint16:
        x = x/(2**16 - 1)
    else:
        raise TypeError("Unexpected image type: {}".format(x.dtype))
    
    # Align breast to left (find breast direction).
    # Check first 10% of image from left and first 10% from right. The breast
    # is aligned to the side with the highest cumulative intensity.
    l = max(int(x.shape[1]*0.1), 1)
    if x[:,-l:].sum() > x[:,:l].sum():
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
        x = np.fliplr(x)
    
    # Crop out background outside of breast (columns) : start 10% in from the
    # left side since some cases start with an empty border and move left to
    # remove the border, then move right to find the end of the breast.
    # Start crop when mean intensity falls below threshold.
    threshold = 0.02
    threshold_left = 0.1
    l = max(int(x.shape[1]*0.1), 1)
    crop_col_left  = 0
    crop_col_right = x.shape[1]
    for col in range(l, -1, -1):
        if x[:,col].mean() < threshold_left:
            crop_col_left = col
            break
    for col in range(l, x.shape[1]):
        if x[:,col].mean() < threshold:
            crop_col_right = col+1
            break
    
    # Crop out background outside of breast (row) : start at middle, move
    # outward. Start crop when mean intensity falls below threshold.
    threshold = 0.02
    crop_row_top = 0
    crop_row_bot = x.shape[0]
    for row in range(x.shape[0]//2, -1, -1):
        if x[row,:].mean() < threshold:
            crop_row_top = row
            break
    for row in range(x.shape[0]//2, x.shape[0], 1):
        if x[row,:].mean() < threshold:
            crop_row_bot = row+1
            break
    
    # Adjust crop to not crop mask (find mask bounding box).
    if mask is not None:
        slice_row, slice_col = ndimage.find_objects(mask>0, max_label=1)[0]    
        crop_col_left  = min(crop_col_left,  slice_col.start)
        crop_col_right = max(crop_col_right, slice_col.stop)
        crop_row_top   = min(crop_row_top,   slice_row.start)
        crop_row_bot   = max(crop_row_bot,   slice_row.stop)
    
    # Apply crop (unless image is reduced to less than 10% of its side 
    # on either axis).
    if (    crop_col_right-crop_col_left > 0.1*x.shape[1]
        and crop_row_bot  -crop_row_top  > 0.1*x.shape[0]):
        image = image[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
        if mask is not None:
            mask = mask[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
            return image, mask
    
    return image


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    try:
        # Convert raw cases to tiff.
        path_processed = args.path_processed
        if path_processed is None:
            path_processed = tempfile.mkdtemp(dir=dir_create)
        convert_raws(read_from=args.path_raws,
                     write_to=path_processed,
                     clip_noise=args.clip_noise,
                     force=args.force)
        
        # Prepare dataset.
        prepare_data_ddsm(args, path_processed)
    except:
        raise
    finally:
        # Clean up temporary path.
        if args.path_processed is None:
            shutil.rmtree(path_processed)
    
    
