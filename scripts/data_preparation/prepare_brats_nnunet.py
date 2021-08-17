###
# Convert an hdf5 brain dataset for use with nnunet, following the data
# conversion approach of the nnunet repo.
###


import argparse
from collections import OrderedDict
import json
import os

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from data_tools.wrap import multi_source_array
from utils.data.brats import prepare_data_brats17


def parse():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--hdf5_from', type=str, required=True,
                        help='URL to directory containing hgg.h5 and lgg.h5 '
                             'brain data prepared by '
                             '`prepare_brats_data_hemispheres.py`.')
    parser.add_argument('--save_to', type=str, required=True,
                        help='URL to directory that shall hold the converted '
                             'data.')
    parser.add_argument('--task_number', type=int, required=True)
    parser.add_argument('--name', type=str, required=True,
                        help='The name to give the converted dataset.')
    parser.add_argument('--data_fraction', type=float, default=1,
                        help='The fraction in (0, 1] of the data to include '
                             'in the training set.')
    parser.add_argument('--data_seed', type=int, default=0,
                        help='The random seed used to split the data into '
                             'training, validation, and test sets.')
    return parser.parse_args()


def convert_data(hdf5_from, save_to, task_number, name,
                 data_fraction=1, data_seed=0):
    # Memory map data and split into training, validation, testing.
    data = prepare_data_brats17(path_hgg=os.path.join(hdf5_from, 'hgg.h5'),
                                path_lgg=os.path.join(hdf5_from, 'lgg.h5'),
                                masked_fraction=1-data_fraction,
                                drop_masked=True,
                                rng=np.random.RandomState(data_seed))
    
    # Merge training and validation splits since nnUnet will do 
    # cross-validation.
    s_train = multi_source_array([data['train']['s'], data['valid']['s']])
    m_train = multi_source_array([data['train']['m'], data['valid']['m']])
    
    # Create save_to directory if it doesn't exist.
    save_path = os.path.join(save_to, f'Task{task_number:03}_{name}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save the data as nii.gz as expected by nnUnet.
    print('Converting training set (union of [train, valid] for '
          'cross-validation).')
    for i, (s, m) in enumerate(tqdm(zip(s_train, m_train),
                                    total=len(s_train))):
        _convert_datapoint(i, s, m, save_path, suffix='Tr')
    print('Converting testing set.')
    for i, (s, m) in enumerate(tqdm(zip(data['test']['s'], data['test']['m']),
                                    total=len(data['test']['s']))):
        _convert_datapoint(i, s, m, save_path, suffix='Ts')
    
    # Create json dataset description as expected by nnUnet.
    n_train = len(s_train)
    n_test = len(data['test']['s'])
    json_dict = OrderedDict()
    json_dict['name'] = name
    json_dict['description'] = 'nothing'
    json_dict['tensorImageSize'] = '4D'
    json_dict['reference'] = 'see BraTS2018'
    json_dict['licence'] = 'see BraTS2019 license'
    json_dict['release'] = '0.0'
    json_dict['modality'] = {
        '0': 'T1',
        '1': 'T1ce',
        '2': 'T2',
        '3': 'FLAIR'
    }
    json_dict['labels'] = {
        '0': 'background',
        '1': 'tumor',
    }
    json_dict['numTraining'] = n_train
    json_dict['numTest'] = n_test
    json_dict['training'] = [{'image': f'./imagesTr/{i:04}.nii.gz',
                              'label': f'./labelsTr/{i:04}.nii.gz'}
                             for i in range(n_train)]
    json_dict['test'] = [f'./imagesTs/{i:04}.nii.gz' for i in range(n_test)]
    with open(os.path.join(save_path, 'dataset.json'), 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


def _convert_datapoint(idx, s, m, save_path, suffix):
    '''
    Expected structure:
    Task500_2DBrainTumour/
    ├── dataset.json
    ├── imagesTr
    ├── imagesTs
    ├── labelsTr
    └── labelsTs
    '''
    assert suffix in ['Tr', 'Ts']
    
    # Set up directories.
    s_dir = os.path.join(save_path, f'images{suffix}')
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)
    m_dir = os.path.join(save_path, f'labels{suffix}')
    if not os.path.exists(m_dir):
        os.makedirs(m_dir)
    
    # Save as compressed nifti.
    for modality in range(s.shape[0]):         # Modality.
        fn = f'{idx:04}_{modality:04}.nii.gz'
        s_write = s[modality:modality+1].astype(np.float32)
        im_sitk = sitk.GetImageFromArray(s_write)
        sitk.WriteImage(im_sitk, os.path.join(s_dir, fn))
    fn = f'{idx:04}.nii.gz'
    m_write = m.astype(bool).astype(np.uint8)
    im_sitk = sitk.GetImageFromArray(m_write)
    sitk.WriteImage(im_sitk, os.path.join(m_dir, fn))


if __name__=='__main__':
    args = parse()
    assert args.task_number >= 500      # Custom tasks start at 500.
    assert args.data_fraction > 0
    assert args.data_fraction <= 1
    convert_data(args.hdf5_from,
                 args.save_to,
                 args.task_number,
                 args.name,
                 args.data_fraction,
                 args.data_seed)
