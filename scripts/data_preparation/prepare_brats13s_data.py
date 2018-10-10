"""
Based on script by Joseph Cohen (Copyright 2018).
Modified by Chris Beckham, Eugene Vorontsov (Copyright 2018).
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
    return parser.parse_args()

args = parse_args()
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def get_data(glob_pattern, is_labels=False):
    data = {}
    c = 0
    for filename in glob.iglob(glob_pattern):
        if "N4ITK" in filename:
            continue    # skip
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
    return data

def normalize_data(data):
    for key in data.keys():
        out = data[key].copy().astype(np.float32)
        mask = out!=0
        out -= out.mean()
        out[mask] -= out[mask].mean()
        out[mask] /= out[mask].std()*5
        data[key] = out

def get_labels(side):
    met = {}
    def count(*classes):
        return sum([np.count_nonzero(side==c) for c in classes])
    met["brain"] = 1.-float(count(0))/np.prod(side.shape)
    met["tumor"] = float(count(4,5))/(count(0)+1e-10)     # Lesion is 4, 5.
    met["has_enough_brain"] = met["brain"] > args.b_thresh
    met["has_tumor"]        = met["tumor"] > args.t_thresh
    return met

for grade in ['HG', 'LG']:

    print("Processing grade: %s" % grade)

    dat = {}
    basepath = os.path.join(args.data_dir, "Synthetic_Data")
    dat['flair'] = get_data(basepath+"/{}/*/*/*Flair.*.mha".format(grade))
    dat['t1'] = get_data(basepath+"/{}/*/*/*T1.*.mha".format(grade))
    dat['t1c'] = get_data(basepath+"/{}/*/*/*T1c.*.mha".format(grade))
    dat['t2'] = get_data(basepath+"/{}/*/*/*T2.*.mha".format(grade))
    for key in dat.keys():
        normalize_data(dat[key])
    dat['labels'] = get_data(basepath+"/{}/*/*/*5more*.mha".format(grade),
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
