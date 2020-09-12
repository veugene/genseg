import argparse
from collections import (defaultdict,
                         OrderedDict)
from functools import partial
import multiprocessing
import os
import shutil
import re
import tempfile

import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage.morphology import (binary_opening,
                                flood)
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


def _process_healthy(im_path, newsize):
    im = sitk.ReadImage(im_path)
    im_np = sitk.GetArrayFromImage(im)
    im_np = trim(im_np)
    im_np = resize(im_np,
                   size=(newsize, newsize),
                   interpolator=sitk.sitkLinear)
    return im_np


def _process_sick(root, ddsm_by_cbis, mask_by_cbis, newsize):
    # Load image.
    im = sitk.ReadImage(ddsm_by_cbis[root])
    im_np = sitk.GetArrayFromImage(im)
    
    # Load all usable masks, merge them together, resize.
    mask_np = None
    for path in mask_by_cbis[root]:
        m = sitk.ReadImage(path)
        m_np = np.squeeze(sitk.GetArrayFromImage(m))
        if mask_np is None:
            mask_np = m_np
        else:
            mask_np = np.logical_or(m_np, mask_np).astype(np.uint8)
    im_np, mask_np = trim(im_np, mask_np)
    im_np = resize(im_np,
                   size=(newsize, newsize, 1),
                   interpolator=sitk.sitkLinear)
    im_np = np.squeeze(im_np)
    mask_np = resize(mask_np,
                     size=(newsize, newsize),
                     interpolator=sitk.sitkNearestNeighbor)
    
    return im_np, mask_np


def _match_masks_to_images(root, masks_by_dir, files_ddsm_s):
    common_match = None
    masks = []
    for path, x, y in masks_by_dir[root]:
        try:
            match_ddsm = files_ddsm_s[x][y]
        except KeyError:
            continue    # Skip
        
        # Select only those with the same view.
        candidates = []     # Tuples : DDSM image, CBIS-DDSM mask.
        view = re.search("(LEFT_CC|LEFT_MLO|RIGHT_CC|RIGHT_MLO)",
                            root).group(0)
        for match in match_ddsm:
            if re.search(view, match):
                candidates.append(match)
        
        # For each candidate image, check the overlap of the mask with the
        # original DDSM mask corresponding to the image. Choose the image
        # with the highest overlap > 0.
        best_candidate = None
        max_overlap = 0
        for match in candidates:
            original_mask_path = match.replace(".tif", "_mask.tif")
            if not os.path.exists(original_mask_path):
                # DDSM image has no mask; skip.
                continue
            mask_orig = sitk.ReadImage(original_mask_path)
            mask_cbis = sitk.ReadImage(path)
            overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_filter.Execute(mask_orig, mask_cbis[:,:,0])
            jaccard = overlap_filter.GetJaccardCoefficient()
            if jaccard > max_overlap:
                max_overlap = jaccard
                best_candidate = match
        if best_candidate is not None:
            if common_match is None:
                common_match = best_candidate
            if common_match!=best_candidate:
                raise AssertionError(
                    "All masks in the same case should match the same "
                    "image in the DDSM set but at least two masks match "
                    "different images for case {}.".format(root))
            masks.append(path)      # This is a usable mask.
    
    # Return values.
    no_match = False
    ddsm_by_cbis = {}
    mask_by_cbis = {}
    if common_match is None:
        no_match = True
    else:
        ddsm_by_cbis[root] = common_match
        mask_by_cbis[root] = masks
    
    return no_match, root, ddsm_by_cbis, mask_by_cbis


def prepare_data_ddsm(args, path_processed):
    
    # Create HDF5 file to save dataset into.
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
    
    # Index all processed DDSM files. For sick cases, store path indexed 
    # by image size. For healthy cases, simply store a list of paths.
    print("Indexing DDSM files.")
    files_ddsm_s = {}
    files_ddsm_s_list = []
    files_ddsm_h = []
    for root, dirs, files in os.walk(os.path.join(path_processed)):
        for fn in files:
            if not fn.endswith(".tif"):
                # Not a processed image; skip.
                continue
            if fn.endswith("mask.tif"):
                # Mask, not image; skip.
                continue
            if root.startswith(os.path.join(path_processed, "normals")):
                # Healthy case; store in a list.
                files_ddsm_h.append(os.path.join(root, fn))
                continue
            files_ddsm_s_list.append(os.path.join(root, fn))
    for path in tqdm(files_ddsm_s_list):
        # Sick case; index.
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        x, y = reader.GetSize()
        if x not in files_ddsm_s:
            files_ddsm_s[x] = {}
        if y not in files_ddsm_s[x]:
            files_ddsm_s[x][y] = []
        files_ddsm_s[x][y].append(path)
    
    # Collect and store healthy cases (resized).
    print("Processing normals (healthy) from DDSM")
    with multiprocessing.Pool() as pool:
        iterator = pool.imap(partial(_process_healthy,
                                     newsize=args.resize),
                             files_ddsm_h)
        for im_np in tqdm(iterator, total=len(files_ddsm_h), smoothing=0.1):
            writer['train']['h'].buffered_write(im_np)
    writer['train']['h'].flush_buffer()
    
    # Identify all masks and their shapes.
    print("Indexing CBIS-DDSM masks.")
    paths_mask = []
    for d in sorted(os.listdir(args.path_cbis)):
        d_path = os.path.join(args.path_cbis, d)
        if not os.path.isdir(d_path):
            # Not a directory; skip.
            continue
        if re.search("^Mass-(Training|Test).*[1-9]$", d):
            for root, dirs, files in os.walk(d_path):
                for fn in files:
                    if fn.endswith('.dcm'):
                        path = os.path.join(root, fn)
                        paths_mask.append(path)
    masks_by_dir = defaultdict(list)
    for path in tqdm(paths_mask):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        x, y, z = reader.GetSize()
        assert(z==1)
        # root omits the number (eg. CC_1, MLO_3 --> CC, MLO)
        root = re.search("^.*Mass-(Training|Test).*(CC|MLO)", path).group(0)
        masks_by_dir[root].append((path, x, y))
    
    # Match sick case masks from CBIS-DDSM to images in DDSM. Matching is done
    # first by image size, then by view (left/right, cc/mlo), and finally
    # by overlaying the mask with the original DDSM mask. Some sick images do 
    # not have corresponding masks in DDSM and so they are ignored here.
    print("Matching CBIS-DDSM masks to DDSM images.")
    ddsm_by_cbis = {}
    mask_by_cbis = {}
    no_match = []
    with multiprocessing.Pool() as pool:
        iterator = pool.imap(partial(_match_masks_to_images,
                                     masks_by_dir=masks_by_dir,
                                     files_ddsm_s=files_ddsm_s),
                             sorted(masks_by_dir.keys()))
        for result in tqdm(iterator, total=len(masks_by_dir.keys()),
                           smoothing=0.1):
            _no_match, _root, _ddsm_by_cbis, _mask_by_cbis = result
            if _no_match:
                no_match.append(_root)
            else:
                ddsm_by_cbis.update(_ddsm_by_cbis)
                mask_by_cbis.update(_mask_by_cbis)
    
    # Report failed cases.
    if len(no_match):
        print("Skipped {} of {} masks since no matching image could be found: "
              "\n{}".format(len(no_match),
                            len(masks_by_dir.keys()),
                            "\n".join(no_match)))
    
    # Sort cases into training, validation, and test subsets.
    train = []
    test  = []
    for k in sorted(mask_by_cbis.keys()):
        if re.search("Mass-Training", k):
            train.append(k)
        elif re.search("Mass-Test", k):
            test.append(k)
        else:
            raise AssertionError("This error is not possible.")
    rng = np.random.RandomState(0)
    rng.shuffle(train)
    n_valid = int(args.validation_fraction*len(train))
    cbis_dirs = {'train': train[n_valid:],
                 'valid': train[:n_valid],
                 'test' : test}
    
    # For each case, combine all lesions into one mask; process image 
    # and mask and store in HDF5 file.
    for fold in ['train', 'valid', 'test']:
        print("Processing cases with masses (sick) : '{}' fold"
              "".format(fold))
        with multiprocessing.Pool() as pool:
            iterator = pool.imap(partial(_process_sick,
                                         ddsm_by_cbis=ddsm_by_cbis,
                                         mask_by_cbis=mask_by_cbis,
                                         newsize=args.resize),
                                 sorted(cbis_dirs[fold]))
            for im_np, mask_np in tqdm(iterator, total=len(cbis_dirs[fold]),
                                       smoothing=0.1):
                writer[fold]['s'].buffered_write(im_np)
                writer[fold]['m'].buffered_write(mask_np)
        writer[fold]['s'].flush_buffer()
        writer[fold]['m'].flush_buffer()


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
    
    # Normalize to within [0, 1] and threshold to binary.
    assert image.dtype==np.uint16
    x = image.copy()
    x_norm = x.astype(np.float)/(2**16 - 1)
    x[x_norm>=0.1] = 1
    x[x_norm< 0.1] = 0
    
    # Align breast to left (find breast direction).
    # Check first 10% of image from left and first 10% from right. The breast
    # is aligned to the side with the highest cumulative intensity.
    l = max(int(x.shape[1]*0.1), 1)
    if x[:,-l:].sum() > x[:,:l].sum():
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
        x = np.fliplr(x)
        x_norm = np.fliplr(x_norm)
    
    # Crop out bright band on left, if present. Use the normalized image
    # instead of the thresholded image in order to differentiate the bright
    # band from breast tissue. Start 20% in from the left side since some 
    # cases start with an empty border and move left to remove the border 
    # (empty or bright).
    t_high = 0.75
    t_low  = 0.1
    start_col_left = max(int(x.shape[1]*0.2), 1)
    crop_col_left  = 0
    for col in range(start_col_left, -1, -1):
        mean = x_norm[:,col].mean()
        if mean < t_low:
            # empty
            crop_col_left = col
            break
        if mean > t_high:
            # bright
            crop_col_left = col
            break
    
    # Crop out other bright bands using thresholded image. Some bright bands
    # are not fully saturated, so it's best to remove them after this 
    # thresholding. This would not work well for the left edge since some
    # breasts take the full edge. For each edge (right, top, bottom), start
    # 20% in and move out. Stop when the mean intensity switches from below
    # the threshold to above the threshold; if it starts above the threshold,
    # then it just means the breast is spanning the entire width/height at 
    # the start position.
    t_high = 0.95
    start_col_right = max(int(x.shape[1]*0.8), 1)
    start_row_top   = max(int(x.shape[0]*0.2), 1)
    start_row_bot   = max(int(x.shape[0]*0.8), 1)
    crop_col_right = x.shape[1]
    crop_row_top   = 0
    crop_row_bot   = x.shape[0]
    prev_mean_col_right  = 1
    prev_mean_row_top    = 1
    prev_mean_row_bot    = 1
    for col in range(start_col_right, x.shape[1]):
        mean = x[:,col].mean()
        if mean > t_high and prev_mean_col_right < t_high:
            crop_col_right = col+1
            break
        prev_mean_col_right = mean
    for row in range(start_row_top, -1, -1):
        mean = x[row,:].mean()
        if mean > t_high and prev_mean_row_top < t_high:
            crop_row_top = row
            break
        prev_mean_row_top = mean
    for row in range(start_row_bot, x.shape[0]):
        mean = x[row,:].mean()
        if mean > t_high and prev_mean_row_bot < t_high:
            crop_row_bot = row+1
            break
        prev_mean_row_top = mean
    
    # Store crop indices for edges - to be used to make sure that the final 
    # crop does not include the edges.
    crop_col_left_edge  = crop_col_left
    crop_col_right_edge = crop_col_right
    crop_row_top_edge   = crop_row_top
    crop_row_bot_edge   = crop_row_bot
    
    # Flood fill breast in order to then crop background out. Start flood 
    # at pixel 20% right from the left edge, center row. Apply the flood to 
    # a the image with the edges cropped out in order to avoid flooding the
    # edges and any artefacts that overlap the edges.
    x_view = x[crop_row_top:crop_row_bot,crop_col_left:crop_col_right]
    flood_row = x_view.shape[0]//2
    flood_col = int(x_view.shape[1]*0.2)
    x_fill = binary_opening(x_view, selem=np.ones((5,5)))
    m_view = flood(x_fill, (flood_row, flood_col), connectivity=1)
    m = np.zeros(x.shape, dtype=np.bool)
    m[crop_row_top:crop_row_bot,crop_col_left:crop_col_right] = m_view
    
    # Crop out background outside of breast. Start 20% in from the left side
    # at the center row and move outward until the columns or rows are empty.
    # For every row or column mean, ignore 20% of each end of the vector in 
    # order to avoid problematic edges.
    frac_row = int(m.shape[0]*0.2)
    frac_col = int(m.shape[1]*0.2)
    for col in range(flood_col, m.shape[1]):
        if not np.any(m[frac_row:-frac_row,col]):
            crop_col_right = min(crop_col_right, col+1)
            break
    for row in range(flood_row, -1, -1):
        if not np.any(m[row,frac_col:-frac_col]):
            crop_row_top = max(crop_row_top, row)
            break
    for row in range(flood_row, m.shape[0]):
        if not np.any(m[row,frac_col:-frac_col]):
            crop_row_bot = min(crop_row_bot, row+1)
            break
    
    # Make sure crop row and column numbers are in range.
    crop_col_right = min(crop_col_right, image.shape[1])
    crop_row_top = max(crop_row_top, 0)
    crop_row_bot = min(crop_row_bot, image.shape[0])
    
    # Adjust crop to not crop mask (find mask bounding box).
    if mask is not None:
        slice_row, slice_col = ndimage.find_objects(mask>0, max_label=1)[0]    
        crop_col_left  = min(crop_col_left,  slice_col.start)
        crop_col_right = max(crop_col_right, slice_col.stop)
        crop_row_top   = min(crop_row_top,   slice_row.start)
        crop_row_bot   = max(crop_row_bot,   slice_row.stop)
    
    # Expand crop 5% of image dim in each direction (right, top, bottom)
    # about the breast in order to avoid clipping any breast.
    crop_col_right = min(int(crop_col_right+x.shape[1]*0.05),
                         crop_col_right_edge)
    crop_row_top   = max(int(crop_row_top-x.shape[0]*0.05),
                         crop_row_top_edge)
    crop_row_bot   = min(int(crop_row_bot+x.shape[0]*0.05),
                         crop_row_bot_edge)
    
    # Apply crop.
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
        dir_create = os.path.dirname(args.path_create)
        if not os.path.exists(dir_create):
            os.makedirs(dir_create)
        path_processed = args.path_processed
        if path_processed is None:
            path_processed = tempfile.mkdtemp(dir=dir_create)
        print("Converting raw DDSM to tiff.")
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
    
    
