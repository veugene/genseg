from __future__ import print_function
from argparse import ArgumentParser
import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import util

def get_kaggle_folder_iterators(batch_size, labels=(0,4), postfix='.jpeg'):
    """Get iterators corresponding to sick and healthy images of the 256px Kaggle images"""
    if "DR_FOLDERS" not in os.environ:
        raise Exception("DR_FOLDERS not found in environent variables. Please source env.sh!")
    fnames0, fnames1 = [], []
    img_folder = "%s/%s" % (os.environ["DR_FOLDERS"], "train-trim-256")
    labels_file = "%s/%s" % (os.environ["DR_FOLDERS"], "trainLabels.csv")
    with open(labels_file) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split(",")
            fname, cls = "%s%s" % (line[0], postfix), int(line[1])
            if cls == labels[0]:
                fnames0.append(fname)
            elif cls == labels[1]:
                fnames1.append(fname)
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(fnames0)
    rnd_state.shuffle(fnames1)
    fnames0_train = fnames0[0:int(0.9*len(fnames0))]
    fnames0_valid = fnames0[int(0.9*len(fnames0))::]
    fnames1_train = fnames1[0:int(0.9*len(fnames1))]
    fnames1_valid = fnames1[int(0.9*len(fnames1))::]
    trs = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    ds_train_a = util.DatasetFromFolder(img_folder, images=fnames0_train, resize_scale=286,
                                        crop_size=256, fliplr=True, transform=trs)
    ds_train_b = util.DatasetFromFolder(img_folder, images=fnames1_train, resize_scale=286,
                                        crop_size=256, fliplr=True, transform=trs)
    ds_train_both = util.MultipleDataset(ds_train_a, ds_train_b)
    train_loader = DataLoader(ds_train_both, batch_size=batch_size, shuffle=True)
    ds_valid_a = util.DatasetFromFolder(img_folder, images=fnames0_valid, resize_scale=286,
                                        crop_size=256, fliplr=True, transform=trs)
    ds_valid_b = util.DatasetFromFolder(img_folder, images=fnames1_valid, resize_scale=286,
                                        crop_size=256, fliplr=True, transform=trs)
    ds_valid_both = util.MultipleDataset(ds_valid_a, ds_valid_b)
    valid_loader = DataLoader(ds_valid_both, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader

def get_idrid_seg_folder_iterators(batch_size, img_size=256):
    """Read from `no_apparent_retinopathy` and `apparant_retinopathy` folders."""
    if "DR_FOLDERS" not in os.environ:
        raise Exception("DR_FOLDERS not found in environent variables. Please source env.sh!")
    ds_sick = util.DatasetFromFolder(
        os.environ["DR_FOLDERS"] + "/idrid_sc1_train/apparent_retinopathy",
        crop_size=img_size, resize_scale=(4288//3, 2848//3),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: (img-0.5)/0.5)])
    )
    ds_healthy = util.DatasetFromFolder(
        os.environ["DR_FOLDERS"] + "/idrid_sc1_train/no_apparent_retinopathy",
        crop_size=img_size, resize_scale=(4288//3, 2848//3),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: (img-0.5)/0.5)])
    )
    ds_both = util.MultipleDataset(ds_sick, ds_healthy)
    loader = DataLoader(ds_both, batch_size=batch_size, shuffle=True)
    return loader
