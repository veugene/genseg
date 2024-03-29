import argparse
import os
import queue
import threading
from collections import OrderedDict

import h5py
import numpy as np
import scipy.misc
import SimpleITK as sitk

from scipy import ndimage

from scripts.data_preparation.prepare_brats_data_hemispheres import thread_pool_executor


def parse():
    parser = argparse.ArgumentParser(description="Prepare BRATS data. Loads "
                                                 "BRATS 2017 data and stores volume slices in an HDF5 archive. "
                                                 "Slices are organized as a group per patient, containing three "
                                                 "groups: \'sick\', \'healthy\', and \'segmentations\'. Sick cases "
                                                 "contain any anomolous class, healthy cases contain no anomalies, and "
                                                 "segmentations are the segmentations of the anomalies. Each group "
                                                 "contains subgroups for each of the three orthogonal planes along "
                                                 "which slices are extracted. For each case, MRI sequences are stored "
                                                 "in a single volume, indexed along the first axis in the following "
                                                 "order: flair, t1ce, t1, t2. All slices are cropped to the minimal "
                                                 "bounding box containing the brain.")
    parser.add_argument('--data_dir',
                        help="The directory containing the BRATS 2017 data. "
                             "Either HGG or LGG.",
                        required=True, type=str)
    parser.add_argument('--save_to',
                        help="Path to save the HDF5 file to.",
                        required=True, type=str)
    parser.add_argument('--skip_bias_correction',
                        help="Whether to skip N4 bias correction.",
                        required=False, action='store_false')
    parser.add_argument('--no_crop',
                        help="Whether to not crop slices to the minimal "
                             "bounding box containing the brain.",
                        required=False, action='store_false')
    parser.add_argument('--min_tumor_fraction',
                        help="Minimum amount of tumour per slice in [0, 1].",
                        required=False, type=float, default=0.01)
    parser.add_argument('--min_brain_fraction',
                        help="Minimum amount of brain per slice in [0, 1].",
                        required=False, type=float, default=0.05)
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    parser.add_argument('--save_debug_to',
                        help="Save images of each slice to this directory, "
                             "for inspection.",
                        required=False, type=str, default=None)
    return parser.parse_args()


def data_loader(data_dir, crop=True):
    tags = ['flair', 't1ce', 't1', 't2', 'seg']
    for dn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dn)

        # Match filenames to tags.
        # NOTE: t1ce chosen when both t1 and t1ce matched
        fn_dict = OrderedDict()
        for fn in sorted(os.listdir(path)):
            match = [t for t in tags if t in fn]
            fn_dict[match[0]] = fn

        # Load files.
        vol_all = []
        segmentation = None
        size = None
        for t in tags:
            vol = sitk.ReadImage(os.path.join(path, fn_dict[t]))
            vol_np = sitk.GetArrayFromImage(vol)

            if size is None:
                size = vol_np.shape
            if vol_np.shape != size:
                raise Exception("Expected {} to have a size of {} but got {}."
                                "".format(fn_dict[t], size, vol_np.shape))

            if t == 'seg':
                segmentation = vol_np.astype(np.int64)
                segmentation = np.expand_dims(segmentation, 0)
            else:
                vol_np = vol_np.astype(np.float32)
                vol_all.append(np.expand_dims(vol_np, 0))

        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=0)

        # Crop to volume.
        if crop:
            bbox = ndimage.find_objects(volume != 0)[0]
            volume = volume[bbox]
            segmentation = segmentation[bbox]

        yield volume, segmentation, dn


def get_slices(volume, segmentation, brain_mask, indices,
               min_tumor_fraction, min_brain_fraction):
    assert 0 <= min_tumor_fraction <= 1
    assert 0 <= min_brain_fraction <= 1

    # Axis transpose order.
    axis = 1
    order = [1, 0, 2, 3]
    volume = volume.transpose(order)
    segmentation = segmentation.transpose(order)
    brain_mask = brain_mask.transpose(order)

    # Select slices. Slices with anomalies are sick, others are healthy.
    indices_anomaly = []
    indices_healthy = []
    for i in range(len(volume)):
        count_total = np.product(volume[i].shape)
        count_brain = np.count_nonzero(brain_mask[i])
        count_tumor = np.count_nonzero(segmentation[i])
        if count_brain == 0:
            continue
        tumor_fraction = count_tumor / float(count_brain)
        brain_fraction = count_brain / float(count_total)
        # print('Brain Fraction: {}, Min Brain Fraction: {}, Tumor Fraction: {}, Min Tumor Fraction: {}'
        #       .format(brain_fraction,
        #               min_brain_fraction,
        #               tumor_fraction,
        #               min_tumor_fraction))
        if brain_fraction > min_brain_fraction:
            if tumor_fraction > min_tumor_fraction:
                indices_anomaly.append(i)
            if count_tumor == 0:
                # print('indices append!!!, count_tumor: {}'.format(count_tumor))
                indices_healthy.append(i)

    # Sort slices.
    slices_dict = OrderedDict()
    slices_dict['healthy'] = volume[indices_healthy]
    slices_dict['sick'] = volume[indices_anomaly]
    slices_dict['segmentation'] = segmentation[indices_anomaly]
    slices_dict['h_indices'] = indices[indices_healthy]
    slices_dict['s_indices'] = indices[indices_anomaly]

    return slices_dict

def preprocess(volume, segmentation, skip_bias_correction=False):
    volume_out = volume.copy()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 10])
    seg_sitk = sitk.GetImageFromArray(segmentation[0])
    for i, vol in enumerate(volume):
        # N4 bias correction.
        if not skip_bias_correction:
            vol_sitk = sitk.GetImageFromArray(vol)
            out_sitk = corrector.Execute(vol_sitk, seg_sitk)
            out = sitk.GetArrayFromImage(out_sitk)
        else:
            out = vol.copy()
        volume_out[i] = out

    # Mean center and normalize by std.
    brain_mask = volume_out != 0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std() * 5  # fit in tanh

    # Get slice indices, with 0 at the center.
    # brain_mask_ax1 = brain_mask.sum(axis=(0,2,3))>0
    # idx_min = np.argmax(brain_mask_ax1)
    # idx_max = len(brain_mask_ax1)-1-np.argmax(np.flipud(brain_mask_ax1))
    # idx_mid = (idx_max-idx_min)//2
    # a = idx_mid-len(brain_mask_ax1)
    # b = len(brain_mask_ax1)+a-1
    # indices = np.arange(a, b)
    indices = np.arange(brain_mask.shape[1])

    # Split volume along hemispheres.
    # mid0 = volume.shape[-1] // 2
    # mid1 = mid0
    # if volume.shape[-1] % 2:
    #     mid0 += 1
    # indices = np.concatenate([indices, indices])
    print("volume_out: {}, segmentation: {}, brain_mask: {}, indices: {}".format(volume_out.shape, segmentation.shape,
                                                                                 brain_mask.shape, indices.shape))
    return volume_out, segmentation, brain_mask, indices


def process_case(case_num, h5py_file, volume, segmentation, fn,
                 min_tumor_fraction, min_brain_fraction,
                 skip_bias_correction=False, save_debug_to=None):
    print("Processing case {}: {}".format(case_num, fn))
    group_p = h5py_file.create_group(str(case_num))
    # TODO: set attribute containing fn.
    vol, seg, m, indices = preprocess(volume, segmentation,
                                      skip_bias_correction)
    slices = get_slices(vol, seg, m, indices,
                        min_tumor_fraction, min_brain_fraction)
    for key in slices.keys():
        print('key: {}, len: {}'.format(key, len(slices[key])))
        if len(slices[key]) == 0:
            kwargs = {}
        else:
            kwargs = {'chunks': (1,) + slices[key].shape[1:],
                      'compression': 'lzf'}
        group_p.create_dataset(key,
                               shape=slices[key].shape,
                               data=slices[key],
                               dtype=slices[key].dtype,
                               **kwargs)

    # Debug outputs for inspection.
    if save_debug_to is not None:
        for key in slices.keys():
            if "indices" in key:
                continue
            dest = os.path.join(save_debug_to, key)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in range(len(slices[key])):
                im = slices[key][i]
                for ch, im_ch in enumerate(im):
                    scipy.misc.imsave(os.path.join(dest, "{}_{}_{}.png"
                                                         "".format(case_num, i, ch)),
                                      slices[key][i][ch])


class thread_pool_executor(object):
    def __init__(self, max_workers, block=False):
        self.max_workers = max_workers
        if not block:
            self._task_queue = queue.Queue()
        else:
            self._task_queue = queue.Queue(max_workers)
        self._run_thread = None

    def submit(self, func, *args, **kwargs):
        self._task_queue.put([func, args, kwargs])

        if self._run_thread is None:
            self._run_thread = threading.Thread(
                target=thread_pool_executor._run,
                args=(self._task_queue, self.max_workers)
            )
            self._run_thread.daemon = True
            self._run_thread.start()

    @staticmethod
    def _run(task_queue, max_workers):
        threads = {}
        termination_queue = queue.Queue()
        for i in range(max_workers):
            termination_queue.put(i)

        def do_task(func, args, kwargs, thread_num, termination_queue):
            func(*args, **kwargs)
            termination_queue.put(thread_num)

        while 1:
            task = task_queue.get()
            if task is None:
                for i in threads:
                    threads[i].join()
                break
            func, args, kwargs = task
            thread_num = termination_queue.get()
            if thread_num in threads:
                threads[thread_num].join()
            threads[thread_num] = threading.Thread( \
                target=do_task,
                args=(func, args, kwargs, thread_num, termination_queue)
            )
            threads[thread_num].daemon = True
            threads[thread_num].start()

    def shutdown(self, wait=False):
        if not wait:
            try:
                while 1:
                    self._task_queue.get()
            except queue.Empty:
                pass
        self._task_queue.put(None)
        if wait:
            self._run_thread.join()

if __name__ == '__main__':
    args = parse()
    # if os.path.exists(args.save_to):
    # raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    try:
        num_threads = args.num_threads
        if num_threads is None:
            num_threads = os.cpu_count()
        executor = thread_pool_executor(max_workers=num_threads, block=True)
        for i, (vol, seg, fn) in enumerate(data_loader(args.data_dir,
                                                       not args.no_crop)):
            executor.submit(process_case, i, h5py_file, vol, seg, fn,
                            args.min_tumor_fraction,
                            args.min_brain_fraction,
                            args.skip_bias_correction,
                            args.save_debug_to)
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        pass
