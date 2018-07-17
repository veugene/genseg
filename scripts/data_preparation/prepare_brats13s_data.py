"""
Based on script by Joseph Cohen (Copyright 2018).
Modified by Chris Beckham (Copyright 2018).
"""

from __future__ import (print_function,
                        division)
import os
import glob
import re
import argparse
import SimpleITK as sitk
import numpy as np
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--b_thresh', type=float, default=0.30)
    parser.add_argument('--t_thresh', type=float, default=0.01)
    parser.add_argument('--both_hemispheres', action='store_true',
                        help='Create examples also from left side of brain')
    parser.add_argument('--max_examples', type=int, default=-1,
                        help='Max examples to process. Use only for debugging')
    parser.add_argument('--normalize', type=str, choices=['max', 'canonical'],
                        default='canonical')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

args = parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def get_data(glob_pattern, is_labels=False):
    data = {}
    c = 0
    for filename in glob.iglob(glob_pattern):
        print(filename)
        path = os.path.normpath(filename).split(os.sep)
        number = list(reversed(path))[2]
        level = list(reversed(path))[3]
        name = level+number

        image_data = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        
        if (is_labels):
            image_data = np.round(image_data) # clean up labels

        if not args.both_hemispheres:
            image_data = image_data[:,:,128:] # crop right side
            data[name] = image_data
        else:
            image_data_left = image_data[:,:,0:128]
            image_data_right = image_data[:,:,128:]
            data[name] = np.vstack((image_data_left,
                                    image_data_right))
        c += 1
        if args.max_examples != -1:
            # If debug flag is enabled, we generate
            # a smaller version of the dataset.
            if c == args.max_examples:
                break
    return data

def normalize_data(data):
    if args.normalize == 'max':
        # Want to norm data into [0,1], and then
        # scale so that it's in range [-2, 2].
        for k,v in data.items():
            data[k] = (((v / np.max(v)) - 0.5) / 0.5) * 2.
    elif args.normalize == 'canonical':
        # Want to norm data by subtracting mean and
        # dividing by standard deviation.
        for k,v in data.items():
            data_nonzero_view = data[k][data[k]>0]
            data[k] -= data_nonzero_view.mean()
            data[k] /= data_nonzero_view.std()
    else:
        raise ValueError("`normalize` should only be 'max' or 'canonical'")

def get_labels(side):
    met = {}
    met["brain"] = (1.*(side!= 0).sum()) / np.prod(side.shape)
    met["tumor"] = (1.*(side > 2).sum()) / ((side != 0).sum() + 1e-10)
    met["has_enough_brain"] = met["brain"] > args.b_thresh
    met["has_tumor"]        = met["tumor"] > args.t_thresh
    if args.debug:
        print("this side metadata:", met)
    return met

for grade in ['HG', 'LG']:

    print("Processing grade: %s" % grade)

    dat = {}
    basepath = os.path.join(args.data_dir, "Synthetic_Data")
    dat['flair'] = get_data(basepath+"/{}/*/*/*Flair.*N4ITK.mha".format(grade))
    dat['t1'] = get_data(basepath+"/{}/*/*/*T1.*N4ITK.mha".format(grade))
    dat['t1c'] = get_data(basepath+"/{}/*/*/*T1c.*N4ITK.mha".format(grade))
    dat['t2'] = get_data(basepath+"/{}/*/*/*T2.*N4ITK.mha".format(grade))
    for key in dat.keys():
        normalize_data(dat[key])
    dat['labels'] = get_data(basepath+"/{}/*/*/*5more*N4ITK.mha".format(grade),
                             is_labels=True)
    labels = dat['labels']
    patients = dat['labels'].keys()

    gd = 'lgg' if grade == 'LG' else 'hgg'
    h5f = h5py.File('%s/%s.h5' % (args.out_dir, gd), 'w')
    counter = {'s': 0, 'h': 0}
    modes = ['flair', 't1', 't1c', 't2']
    for patient in patients:
        h5f.create_group(patient)
        # Now figure out what slices in this
        # patient are sick or healthy.
        h_idxs, s_idxs = [], []
        for b, slice_ in enumerate(labels[patient]):
            met = get_labels(slice_)
            if met['has_enough_brain']:
                if met['has_tumor']:
                    s_idxs.append(b)
                    counter['s'] += 1
                else:
                    h_idxs.append(b)
                    counter['h'] += 1
        healthy_slices = []
        sick_slices = []
        # Get all modes (t1, t2, etc.) and concatenate
        # them on the channel axis.
        for mode in modes:
            healthy_slices.append(dat[mode][patient][h_idxs])
            sick_slices.append(dat[mode][patient][s_idxs])
        # X_slices has shape (n, 4, h, w) now.
        healthy_slices = np.asarray(healthy_slices).swapaxes(0,1)
        sick_slices = np.asarray(sick_slices).swapaxes(0,1)
        masks = labels[patient][s_idxs]
        h5f[patient].create_group('healthy')
        h5f[patient].create_group('sick')
        h5f[patient].create_group('segmentation')
        h5f[patient]['sick'].create_dataset(
            'axis_1', data=sick_slices)
        h5f[patient]['segmentation'].create_dataset(
            'axis_1', data=masks[:,np.newaxis,:,:])
        h5f[patient]['healthy'].create_dataset(
            'axis_1', data=healthy_slices)
    h5f.close()
