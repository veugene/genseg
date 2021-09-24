import argparse
from functools import partial
import multiprocessing
import os

import h5py
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from tqdm import tqdm

NUM_CASES = 130


def get_parser():
    parser = argparse.ArgumentParser(description=""
                                                 "Create an HDF5 dataset with LiTS data by using axial slices with "
                                                 "lesion as sick images and those without lesion as healthy images. "
                                                 "Image normalization is done for each volume by subtracting the mean "
                                                 "of the whole liver and dividing by its resulting standard deviation. "
                                                 "A histogram of the relative position of sick slices within the liver "
                                                 "is computed and values are drawn from this histogram for the healthy "
                                                 "slices so that healthy slices may be sampled during training from "
                                                 "the different positions in the liver at the same frequency that sick "
                                                 "slices appear in those positions."
                                     )
    parser.add_argument('path_data', type=str,
                        help="path to the LiTS directory")
    parser.add_argument('--path_create', type=str,
                        default='./data/lits/lits.h5',
                        help="path to save the prepared HDF5 dataset to")
    parser.add_argument('--min_lesion_width', type=int, default=6,
                        help="a slice is included as a sick slice only if at "
                             "least one lesion has this width or larger in "
                             "both the x and y dimensions")
    parser.add_argument('--min_liver_fraction', type=float, default=0.04,
                        help="a slice is included only if the liver takes up "
                             "at least this fraction of the slice area")
    parser.add_argument('--n_bins', type=int, default=20,
                        help="number of bins to sort axial positions into "
                             "when ensuring that the distribution of liver "
                             "slice axial positions equals the distribution "
                             "of lesion slice axial positions")
    parser.add_argument('--resize', type=int, default=256,
                        help="resize axial slices from 512x512 to a square of "
                             "this size")
    return parser


def index_slices(paths, min_lesion_width=0, min_liver_fraction=0.25):
    vol_path, seg_path = paths

    # Load volume, segmentation.
    vol_sitk = sitk.ReadImage(vol_path)
    vol = sitk.GetArrayFromImage(vol_sitk)
    seg_sitk = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg_sitk)

    # Find liver height in z.
    liver_bounds = ndimage.find_objects(seg == 1)[0]
    liver_extent = liver_bounds[0]  # z slice

    # Record every z index for which the liver takes up at least
    # `min_liver_fraction` of the area.
    n_pixels_slice = np.prod(seg.shape[1:])
    frac_liver = np.sum(seg[liver_extent] > 0, axis=(1, 2)) / n_pixels_slice
    valid_liver_indices = (liver_extent.start
                           + np.where(frac_liver > min_liver_fraction)[0])

    ## If no liver for given `min_liver_fraction`, return.
    # if len(valid_lesion_indices)==0:
    # return slice(None, None), [], []

    # Determine the axial range that tightly contains the valid liver slices.
    valid_liver_extent = slice(valid_liver_indices.min(),
                               valid_liver_indices.max())

    # Record every z index for which there are lesion voxels.
    lesion_indices = np.where(np.sum(seg == 2, axis=(1, 2)) > 0)[0]

    # Identify lesions.
    lesion_map, n_lesions = ndimage.measurements.label(seg == 2, output=np.uint8)

    # Filter out slices that don't have at least one lesion whose size in
    # x and y is larger than `min_lesion_width`. Exclude slices where the
    # liver fraction is too small.
    valid_lesion_indices = []
    for lesion_id in range(1, n_lesions + 1):
        _x = lesion_map[valid_liver_indices] == lesion_id
        _b = np.logical_and(np.any(_x.sum(axis=1) > min_lesion_width, axis=1),
                            np.any(_x.sum(axis=2) > min_lesion_width, axis=1))
        _indices = valid_liver_indices[np.where(_b)[0]]
        valid_lesion_indices.extend(_indices)

    # Exclude lesion slices from the liver.
    valid_liver_indices = sorted(list(
        set(valid_liver_indices).difference(lesion_indices)))

    return valid_liver_extent, valid_liver_indices, valid_lesion_indices


class histogram(object):
    def __init__(self, x, bins):
        self.x = x
        self.bins = bins
        self.values, self.edges = np.histogram(x, bins=bins)
        self.values = self.values / self.values.sum()

    def get_value(self, index, liver_extent):
        # Normalize the index to [0, 1] within the liver and get the
        # histogram value at that normalized index. This code assumes that
        # bins are of equal size.
        liver_height = liver_extent.stop - liver_extent.start
        normalized_index = (index - liver_extent.start) / liver_height
        e0, e1 = self.edges[0], self.edges[-1]
        if normalized_index < e0 or normalized_index > e1:
            value = 0
        elif normalized_index == e1:
            value = self.values[-1]
        else:
            i = int((normalized_index - e0) / (e1 - e0) * self.bins)
            value = self.values[i]
        return value


def compute_position_histogram(liver_extents, lesion_indices, bins=10):
    # Compute normalized indices for lesions as relative positions in [0, 1] in
    # the liver.
    assert len(liver_extents) == len(lesion_indices)
    all_normalized_indices = []
    for extent, indices in zip(liver_extents, lesion_indices):
        normalized_indices = [(idx - extent.start) / (extent.stop - extent.start)
                              for idx in indices]
        all_normalized_indices.extend(normalized_indices)

    # Compute a histogram over these positions.
    return histogram(all_normalized_indices, bins=bins)


def get_slices(inputs, size, hist):
    vol_path, seg_path, liver_extent, indices_h, indices_s = inputs

    # Load volume, segmentation.
    vol_sitk = sitk.ReadImage(vol_path)
    vol = sitk.GetArrayFromImage(vol_sitk)
    seg_sitk = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg_sitk)

    # Canonical normalization : consider only liver tissue. Use statistics
    # from the entire liver.
    vol = (vol - vol[seg == 1].mean()) / (vol[seg == 1].std() * 5 + 1)  # fit in tanh

    # Get axial slices.
    vol_slices_h = vol[indices_h]
    vol_slices_s = vol[indices_s]
    seg_slices_h = seg[indices_h]
    seg_slices_s = seg[indices_s]

    # Resize slices.
    vol_slices_h, seg_slices_h = crop_and_resize(vol_slices_h,
                                                 seg_slices_h,
                                                 size=(size, size))
    vol_slices_s, seg_slices_s = crop_and_resize(vol_slices_s,
                                                 seg_slices_s,
                                                 size=(size, size))

    # Get positional histogram values for slices. These will be used to
    # determine the relative frequency with which healthy slices are sampled
    # during training.
    histogram_values = [hist.get_value(idx, liver_extent) for idx in indices_h]
    histogram_values = np.array(histogram_values)

    # Outputs.
    h = vol_slices_h
    s = vol_slices_s
    m = np.uint8(seg_slices_s == 2)
    return h, s, m, histogram_values


def crop_and_resize(stack_vol, stack_seg, size):
    '''
    Set non-liver to zero, crop to the liver, and then resize to (size, size).
    '''
    out_vol = np.zeros((len(stack_vol),) + size, dtype=stack_vol.dtype)
    out_seg = np.zeros((len(stack_seg),) + size, dtype=stack_seg.dtype)
    for i, (vol_slice, seg_slice) in enumerate(zip(stack_vol, stack_seg)):
        bbox = ndimage.find_objects(seg_slice > 0)[0]
        vol_slice[seg_slice == 0] = 0  # Set background to zero.
        out_vol[i] = resize(vol_slice[bbox],
                            size=size,
                            interpolator=sitk.sitkLinear)
        out_seg[i] = resize(seg_slice[bbox],
                            size=size,
                            interpolator=sitk.sitkNearestNeighbor)
    return out_vol, out_seg


def resize(image, size, interpolator=sitk.sitkLinear):
    sitk_image = sitk.GetImageFromArray(image)
    new_spacing = [x * y / z for x, y, z in zip(
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
    out = sitk.GetArrayFromImage(sitk_out)
    return out


def prepare_dataset(args):
    # Create HDF5 file to save dataset into.
    f = h5py.File(args.path_create, 'w')
    f.create_group('h')
    f.create_group('s')
    f.create_group('m')
    f.create_group('h_histogram')

    # Record filenames.
    paths_vol = [os.path.join(args.path_data,
                              "volume-{}.nii.gz".format(n))
                 for n in range(NUM_CASES)]
    paths_seg = [os.path.join(args.path_data,
                              "segmentation-{}.nii.gz".format(n))
                 for n in range(NUM_CASES)]

    # Index slices.
    print("Indexing data.")
    all_liver_extents = []
    all_liver_indices = []
    all_lesion_indices = []
    with multiprocessing.Pool() as pool:
        iterator = pool.imap(partial(index_slices,
                                     min_lesion_width=args.min_lesion_width,
                                     min_liver_fraction=args.min_liver_fraction),
                             zip(paths_vol, paths_seg))
        for out in tqdm(iterator, total=NUM_CASES):
            liver_extent, liver_indices, lesion_indices = out
            all_liver_extents.append(liver_extent)
            all_liver_indices.append(liver_indices)
            all_lesion_indices.append(lesion_indices)

    # Build histogram of axial position of lesion slices.
    print("Computing lesion positioning histogram.")
    hist = compute_position_histogram(all_liver_extents,
                                      all_lesion_indices,
                                      bins=args.n_bins)

    # Extract and preprocess data. Storing slices as healthy or sick.
    # Inputs have the scan slice and its liver segmentation. Masks are
    # lesion masks.
    print("Extracting data.")
    with multiprocessing.Pool() as pool:
        iterator = pool.imap(partial(get_slices,
                                     size=args.resize,
                                     hist=hist),
                             zip(paths_vol,
                                 paths_seg,
                                 all_liver_extents,
                                 all_liver_indices,
                                 all_lesion_indices))
        for i, out in enumerate(tqdm(iterator, total=NUM_CASES)):
            h, s, m, histogram_values = out
            chunks = (1, args.resize, args.resize)
            if len(h):
                f['h'].create_dataset(str(i),
                                      shape=h.shape,
                                      data=h,
                                      dtype=np.float32,
                                      chunks=chunks,
                                      compression='lzf')
                f['h_histogram'].create_dataset(str(i),
                                                shape=histogram_values.shape,
                                                data=histogram_values,
                                                dtype=np.float32)
            if len(s):
                f['s'].create_dataset(str(i),
                                      shape=s.shape,
                                      data=s,
                                      dtype=np.float32,
                                      chunks=chunks,
                                      compression='lzf')
                assert len(s) == len(m)
                f['m'].create_dataset(str(i),
                                      shape=m.shape,
                                      data=m,
                                      dtype=np.uint8,
                                      chunks=(1, args.resize, args.resize),
                                      compression='lzf')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    try:
        prepare_dataset(args)
    except:
        raise