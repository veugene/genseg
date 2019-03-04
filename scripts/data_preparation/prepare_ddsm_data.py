import argparse
from collections import OrderedDict

import h5py
import numpy as np
from skimage import filters
from skimage import transform
from tqdm import tqdm

from data_tools.io import h5py_array_writer


def get_parser():
    parser = argparse.ArgumentParser(description="DDSM seg.")
    parser.add_argument('--data_from', type=str, default='./data/ddsm/ddsm.h5')
    parser.add_argument('--data_save', type=str,
                        default='./data/ddsm/ddsm_simple.h5')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser


def prepare_data_ddsm(path):
    """
    Given an HDF5 file of DDSM data, create a simpler HDF5 file of image
    patches resized to a constant size and split into training, validation,
    and testing subsets.
    
    path (string) : Path of the h5py file containing the DDSM data.
    
    NOTE: A constant random seed (0) is always used to determine the training/
    validation split.
    
    Returns dictionary: healthy slices, sick slices, and segmentations for 
    the training, validation, and testing subsets.
    """
    
    # Random 20% data split for validation, random 20% for testing.
    # 
    # First, split out a random sample of the sick patients into a
    # validation/test set. Any sick patients in the validation/test set that 
    # reoccur in the healthy set are placed in the validation/test set for 
    # the healthy set, as well. The remainder of the healthy validation/test
    # set is completed with a random sample.
    case_id_h = {}
    case_id_s = {}
    try:
        h5py_file = h5py.File(path, mode='r')
    except:
        print("Failed to open data: {}".format(path))
        raise
    rnd_state = np.random.RandomState(0)
    indices = {'s': rnd_state.permutation(len(h5py_file['sick'].keys())),
               'h': rnd_state.permutation(len(h5py_file['healthy'].keys()))}
    def get(keys, indices): # Index into keys object.
        return [k for i, k in enumerate(keys) if i in indices]
    def split(n_cases):
        indices_s_split = indices['s'][:n_cases]
        case_id_s_split = get(h5py_file['sick'].keys(), indices_s_split)
        indices_s_and_h = [i for i, k in enumerate(h5py_file['healthy'].keys())
                           if k in case_id_s_split]
        indices_h_only  = [i for i, k in enumerate(h5py_file['healthy'].keys())
                           if i not in indices_s_and_h]
        n_random = n_cases-len(indices_s_and_h)
        indices_h_split = indices_s_and_h+indices_h_only[:n_random]
        case_id_h_split = get(h5py_file['healthy'].keys(), indices_h_split)
        indices['s'] = [i for i in indices['s'] if i not in indices_s_split]
        indices['h'] = [i for i in indices['h'] if i not in indices_h_split]
        return case_id_s_split, case_id_h_split
    n_valid = int(0.2*len(indices['s']))
    case_id_s['valid'], case_id_h['valid'] = split(n_valid)
    case_id_s['test'],  case_id_h['test']  = split(n_valid)
    
    # Remaining cases go in the training set. At this point, indices for
    # cases in the validation and test sets have been removed.
    case_id_s['train'] = get(h5py_file['sick'].keys(),    indices['s'])
    case_id_h['train'] = get(h5py_file['healthy'].keys(), indices['h'])
    indices = None  # Clear remaining indices.
    
    # Assemble cases with corresponding segmentations; split train/valid.
    cases_h = {'train': [], 'valid': [], 'test': []}
    cases_s = {'train': [], 'valid': [], 'test': []}
    cases_m = {'train': [], 'valid': [], 'test': []}
    for split in ['train', 'valid', 'test']:
        for case_id in case_id_s[split]:
            f = h5py_file['sick'][case_id]
            for view in f.keys():
                for key in filter(lambda x:x.startswith('abnormality'),
                                  f[view].keys()):
                    cases_s[split].append(f[view][key]['cropped_image'])
                    cases_m[split].append(f[view][key]['cropped_mask'])
        for case_id in case_id_h[split]:
            f = h5py_file['healthy'][case_id]
            for view in f.keys():
                cases_h[split].append(f[view]['patch'])
    
    # Merge all arrays in each list of arrays.
    data = OrderedDict([('train', OrderedDict()),
                        ('valid', OrderedDict()),
                        ('test',  OrderedDict())])
    for key in data.keys():
        data[key]['h'] = _list(cases_h[key], np.uint16)
        data[key]['s'] = _list(cases_s[key], np.uint16)
        data[key]['m'] = _list(cases_m[key], np.uint8)
    return data


class _list(object):
    def __init__(self, indexable, dtype):
        self._items = indexable
        self.dtype = dtype
    def __getitem__(self, idx):
        elem = self._items[idx]
        if elem is not None:
            elem = elem[...]
        return elem
    def __len__(self):
        return len(self._items)


def resize(image, size, order=1):
    image = image.copy()
    out_image = np.zeros(image.shape[:-2]+size, dtype=image.dtype)
    for idx in np.ndindex(image.shape[:-2]):
        if any([a<b for a,b in zip(size, image.shape[-2:])]) and order>0:
            # Downscaling - smooth, first.
            s = [(1-float(a)/b)/2. if a<b else 0
                 for a,b in zip(size, image.shape[-2:])]
            image[idx] = filters.gaussian(image[idx], sigma=s)
        out_image[idx] = transform.resize(image[idx],
                                          output_shape=size,
                                          mode='constant',
                                          order=order,
                                          cval=0,
                                          clip=False,
                                          preserve_range=True)
    return out_image


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    data = prepare_data_ddsm(args.data_from)
    
    # Create hdf5 file and groups.
    f = h5py.File(args.data_save, 'w')
    g_train = f.create_group('train')
    g_valid = f.create_group('valid')
    g_test  = f.create_group('test')
    
    # Create arrays.
    for fold in ['train', 'valid', 'test']:
        for key in ['h', 's', 'm']:
            print("Writing {}/{}".format(fold, key))
            dtype = data[fold][key][0].dtype
            shape = (args.image_size, args.image_size)
            writer = h5py_array_writer(
                data_element_shape=shape,
                dtype=dtype,
                batch_size=args.batch_size,
                filename=args.data_save,
                array_name="{}/{}".format(fold, key),
                append=True,
                kwargs={'chunks': (1,)+shape})
            for case in tqdm(data[fold][key]):
                case_resized = resize(case, size=shape, order=1)
                writer.buffered_write(case_resized)
            writer.flush_buffer()
            