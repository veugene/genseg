from __future__ import (print_function,
                        division)
import os
import argparse
from collections import OrderedDict
#from concurrent.futures import ThreadPoolExecutor
import threading
try:
    import queue            # python 3
except ImportError:
    import Queue as queue   # python 2

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import h5py


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
                        required=False, action='store_true')
    parser.add_argument('--num_threads',
                        help="The number of parallel threads to execute.",
                        required=False, type=int, default=None)
    return parser.parse_args()


def data_loader(data_dir):
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
            
            if t=='seg':
                segmentation = vol_np.astype(np.int64)
                segmentation = np.expand_dims(segmentation, 0)
            else:
                vol_np = vol_np.astype(np.float32)
                vol_all.append(np.expand_dims(vol_np, 0))
        
        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=0)
        
        # Crop to volume.
        bbox = ndimage.find_objects(volume!=0)[0]
        volume = volume[bbox]
        segmentation = segmentation[bbox]
            
        yield volume, segmentation, dn
        
        
def get_slices(volume, segmentation, axis):
    # Axis transpose order.
    order = [axis]+[i for i in range(4) if i!=axis]
    
    # Fill this dict.
    slices_dict = OrderedDict()
    
    # Locate anomalies. These are sick slices.
    indices_anomaly = np.unique(np.where(segmentation)[axis])
    indices_anomaly_slice = [slice(None, None)]*4
    indices_anomaly_slice[axis] = indices_anomaly
    slices_dict['sick'] = volume[indices_anomaly_slice].transpose(order)
    slices_dict['segmentation'] = \
        segmentation[indices_anomaly_slice].transpose(order)

    # All other slices without anomalies are healthy.
    indices_all = range(segmentation.shape[axis])
    indices_healthy = [i for i in indices_all if i not in indices_anomaly]
    indices_healthy_slice = [slice(None, None)]*4
    indices_healthy_slice[axis] = indices_healthy
    slices_dict['healthy'] = volume[indices_healthy_slice].transpose(order)
        
    return slices_dict


def preprocess(volume, segmentation, skip_bias_correction=False):
    volume_out = volume.copy()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50,50,50,10])
    seg_sitk = sitk.GetImageFromArray(segmentation[0])
    for i, vol in enumerate(volume):
        # N4 bias correction.
        if not skip_bias_correction:
            vol_sitk = sitk.GetImageFromArray(vol)
            out_sitk = corrector.Execute(vol_sitk, seg_sitk)
            out = sitk.GetArrayFromImage(out_sitk)
        else:
            out = vol.copy()
        
        # Mean center and normalize by std.
        out_nonzero_view = out[out>0]
        out -= out_nonzero_view.mean()
        out /= out_nonzero_view.std()
        
        volume_out[i] = out
        
    return volume_out


def process_case(case_num, h5py_file, volume, segmentation, fn,
                 skip_bias_correction=False):
    print("Processing case {}: {}".format(case_num, fn))
    group_p = h5py_file.create_group(str(case_num))
    # TODO: set attribute containing fn.
    for axis in [1,2,3]:
        volume = preprocess(volume, segmentation, skip_bias_correction)
        slices = get_slices(volume, segmentation, axis=axis)
        for key in slices.keys():
            group_k = group_p.require_group(key)
            group_k.create_dataset('axis_{}'.format(axis),
                                   shape=slices[key].shape,
                                   data=slices[key],
                                   dtype=slices[key].dtype,
                                   chunks=(1,)+slices[key].shape[1:],
                                   compression='lzf')
                                       
                                       
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
            self._run_thread = threading.Thread(\
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
            threads[thread_num] = threading.Thread(\
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


if __name__=='__main__':
    args = parse()
    #if os.path.exists(args.save_to):
        #raise ValueError("Path to save data already exists. Aborting.")
    h5py_file = h5py.File(args.save_to, mode='w')
    try:
        num_threads = args.num_threads
        if num_threads is None:
            num_threads = os.cpu_count()
        executor = thread_pool_executor(max_workers=num_threads, block=True)
        for i, (vol, seg, fn) in enumerate(data_loader(args.data_dir)):
            executor.submit(process_case, i, h5py_file, vol, seg, fn,
                            args.skip_bias_correction)
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        pass
