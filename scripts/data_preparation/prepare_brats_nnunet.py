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
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument('--hdf5_from', type=str, required=True,
                        help="URL to directory containing hgg.h5 and lgg.h5 "
                             "brain data prepared by "
                             "`prepare_brats_data_hemispheres.py`.")
    parser.add_argument('--save_to', type=str, required=True,
                        help="URL to directory that shall hold the converted "
                             "data.")
    parser.add_argument('--name', type=str, required=True,
                        help="The name to give the converted dataset.")
    parser.add_argument('--data_seed', type=int, default=0,
                        help="The random seed used to split the data into "
                             "training, validation, and test sets.")
    return parser.parse_args()


def convert_data(hdf5_from, save_to, name, data_seed=0):
    # Memory map data and split into training, validation, testing.
    data = prepare_data_brats17(path_hgg=os.path.join(args.data, "hgg.h5"),
                                path_lgg=os.path.join(args.data, "lgg.h5"),
                                rng=np.random.RandomState(data_seed))
    
    # Merge training and validation splits since nnUnet will do 
    # cross-validation.
    s_train = multi_source_array(data['train']['s'], data['valid']['s'])
    m_train = multi_source_array(data['train']['m'], data['valid']['m'])
    
    # Create save_to directory if it doesn't exist.
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    
    # Save the data as nii.gz as expected by nnUnet.
    print("Converting training set (union of [train, valid] for "
          "cross-validation).")
    for s, m in tqdm(zip(s_train, m_train)):
        _convert_datapoint(s, m, save_to, suffix='Tr')
    print("Converting testing set.")
    for s, m in tqdm(zip(data['test']['s'], data['test']['m'])):
        _convert_datapoint(s, m, save_to, suffix='Ts')
    
    # Create json dataset description as expected by nnUnet.
    n_train = len(s_train)
    n_test = len(data['test']['s'])
    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2018"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2018"
    json_dict['licence'] = "see BraTS2019 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "tumor",
    }
    json_dict['numTraining'] = n_train
    json_dict['numTest'] = n_test
    json_dict['training'] = [{'image': f"./imagesTr/{i}.nii.gz",
                              'label': f"./labelsTr/{i}.nii.gz"}
                             for i in range(n_train)]
    json_dict['test'] = [{'image': f"./imagesTs/{i}.nii.gz",
                          'label': f"./labelsTs/{i}.nii.gz"}
                          for i in range(n_test)]
    with open(os.path.join(save_to, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


def _convert_datapoint(s, m, save_to, suffix):
    assert suffix in ['Tr', 'Ts']


if __name__=='__main__':
    args = parse()
    convert_data(args.hdf5_from,
                 args.save_to,
                 args.name,
                 args.data_seed)
