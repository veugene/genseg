import argparse
from collections import (defaultdict,
                         OrderedDict)
import os
import shutil
import re
import tempfile

import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from data_tools.io import h5py_array_writer
from data_tools.wrap import multi_source_array
from ddsm_normals.make_dataset import make_data_set as convert_normals


def get_parser():
    parser = argparse.ArgumentParser(description=""
        "Create an HDF5 dataset with DDSM data by combining the CBIS-DDSM "
        "dataset with normal (healthy) cases from the original DDSM "
        "dataset. All images are cropped to remove irrelevant background "
        "and resized to a square.")
    parser.add_argument('path_cbis', type=str,
                        help="path to the CBIS-DDSM data directory")
    parser.add_argument('path_healthy', type=str,
                        help="path to the DDSM `normals` directory")
    parser.add_argument('--healthy_is_processed', action='store_true',
                        help="`path_healthy` points to a directory that "
                             "contains the processed tiff files rather "
                             "than the original raw files")
    parser.add_argument('--path_create', type=str,
                        default='./data/ddsm/ddsm.h5',
                        help="path to save the HDF5 file to")
    parser.add_argument('--validation_fraction', type=float, default=0.2,
                        help="fraction of (sick, CBIS-DDSM) training data "
                             "to use for the validation subset")
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser


def prepare_data_ddsm(args):
    
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
                                             dtype=np.uint8,
                                             **writer_kwargs)
    for fold in ['train', 'valid', 'test']:
        for key in ['s', 'm']:
            writer[fold][key] = h5py_array_writer(
                array_name='{}/{}'.format(fold, key),
                dtype=np.uint16 if key=='s' else np.uint8,
                **writer_kwargs)
    
    try:
        # Convert raw healthy cases to tiff.
        if args.healthy_is_processed:
            path_temp = args.path_healthy
        else:
            path_temp = tempfile.mkdtemp()
            convert_normals(read_from=args.path_healthy,
                            write_to=path_temp,
                            force=False)
    
        # Collect and store healthy cases (resized).
        print("Processing normals (healthy) from DDSM")
        healthy = []
        for root, dirs, files in os.walk(path_temp):
            healthy.extend([os.path.join(root, fn) for fn in files])
        for im_path in tqdm(healthy):
            if not im_path.endswith('.tif'):
                continue
            im = sitk.ReadImage(im_path)
            im_np = sitk.GetArrayFromImage(im)
            im_np = trim(im_np)
            im_np = resize(im_np,
                           size=(args.resize, args.resize),
                           interpolator=sitk.sitkLinear)
            writer['train']['h'].buffered_write(im_np)
        writer['train']['h'].flush_buffer()
    except:
        raise
    finally:
        # Clean up temporary path.
        if not args.healthy_is_processed:
            shutil.rmtree(path_temp)
    
    # Collect sick cases.
    dirs_image = {'train': [], 'test': []}
    dirs_mask  = {'train': defaultdict(list), 'test': defaultdict(list)}
    for d in sorted(os.listdir(args.path_cbis)):
        if not os.path.isdir(os.path.join(args.path_cbis, d)):
            continue
        if   re.search("^Mass-Training.*(CC|MLO)$", d):
            dirs_image['train'].append(d)
        elif re.search("^Mass-Test.*(CC|MLO)$", d):
            dirs_image['test'].append(d)
        elif re.search("^Mass-Training.*[1-9]$", d):
            key = re.search("^Mass-Training.*(CC|MLO)", d).group(0)
            dirs_mask['train'][key].append(d)
        elif re.search("^Mass-Test.*[1-9]$", d):
            key = re.search("^Mass-Test.*(CC|MLO)", d).group(0)
            dirs_mask['test'][key].append(d)
        else:
            pass
    
    # Split sick training cases into training and validation subsets.
    keys = sorted(dirs_image['train'])
    rng = np.random.RandomState(0)
    rng.shuffle(keys)
    n_valid = int(args.validation_fraction*len(keys))
    keys_valid, keys_train = keys[:n_valid], keys[n_valid:]
    
    # Create new sick cases dicts with validation subset.
    dirs_image_split = {
        'train': keys_train,
        'valid': keys_valid,
        'test': dirs_image['test']
        }
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
        for im_dir in tqdm(sorted(dirs_image_split[fold])):
            mask_dir_list = dirs_mask_split[fold][im_dir]
            
            # Function to get path for every dcm image in a directory tree.
            def get_all_dcm(path):
                dcm_list = []
                for root, dirs, files in os.walk(path):
                    for fn in files:
                        if fn.endswith('.dcm'):
                            dcm_list.append(os.path.join(root, fn))
                return dcm_list
            
            # Load image, resize, and store in HDF5.
            im_path = get_all_dcm(os.path.join(args.path_cbis, im_dir))
            assert len(im_path)==1  # there should only be one dcm file
            im_path = im_path[0]
            im = sitk.ReadImage(im_path)
            im_np = sitk.GetArrayFromImage(im)
            im_np = np.squeeze(im_np)
            im_size = im.GetSize()
            
            # Load all masks, merge them together, resize, and store in HDF5.
            mask_np = None
            for d in mask_dir_list:
                mask_path = get_all_dcm(os.path.join(args.path_cbis, d))
                assert len(mask_path)==2  # there should be two dcm files
                m_np = None
                for path in mask_path:
                    m = sitk.ReadImage(path)
                    if m.GetSize()==im_size:
                        # Of the two files, the mask is the one with the same
                        # image size as the full xray image.
                        m_np = sitk.GetArrayFromImage(m)
                if m_np is None:
                    # Could not find a mask matching the image dimensions!
                    # Don't bother looking at the other masks. Just fail this
                    # case.
                    break
                m_np = np.squeeze(m_np)
                if mask_np is None:
                    mask_np = m_np
                else:
                    mask_np = np.logical_or(m_np, mask_np).astype(np.uint8)
            if mask_np is None:
                # One or more masks does not match the image dimensions.
                # Skip this case and add it to the fail list.
                fail_list.append(im_dir)
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
            crop_col_right = col
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
            crop_row_bot = row
            break
    
    # Apply crop.
    image = image[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
    if mask is not None:
        mask = mask[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
        return image, mask
    
    return image


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    data = prepare_data_ddsm(args)
