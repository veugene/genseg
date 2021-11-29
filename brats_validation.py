import argparse
from collections import OrderedDict
from functools import partial
import imp
import json
import os

import h5py
import numpy as np
import SimpleITK as sitk
import torch

from brats_segmentation import get_parser as get_model_parser
import model.bd_segmentation
from model.bd_segmentation import segmentation_model
from utils.experiment import experiment


def parse():
    parser = argparse.ArgumentParser(description=""
        "Evaluate a BDS3 model on test set volumes. Each volumes is loaded "
        "from the image file. The model is applied to every full slice, not "
        "to every half-slice/hemisphere. Every stack of slices forming a "
        "volume is one batch. No cropping to the brain is done.")
    parser.add_argument('experiment_path', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('save_results_to', type=str)
    parser.add_argument('--model_kwargs', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=80)
    return parser.parse_args()


def preprocess(volume):
    volume_out = volume.copy()
    
    # Mean center and normalize by std.
    brain_mask = volume_out!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh
    
    return volume_out


def data_loader(data_dir):
    tags = ['flair', 't1ce', 't1', 't2']
    for dn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dn)
        if not os.path.isdir(path):
            continue
        
        # Match filenames to tags.
        # NOTE: t1ce chosen when both t1 and t1ce matched
        fn_dict = OrderedDict()
        for fn in sorted(os.listdir(path)):
            match = [t for t in tags if t in fn]
            fn_dict[match[0]] = fn
            
        # Load files.
        vol_all = []
        size = None
        for t in tags:
            vol = sitk.ReadImage(os.path.join(path, fn_dict[t]))
            vol_np = sitk.GetArrayFromImage(vol)
            
            if size is None:
                size = vol_np.shape
            if vol_np.shape != size:
                raise Exception("Expected {} to have a size of {} but got {}."
                                "".format(fn_dict[t], size, vol_np.shape))
            
            vol_np = vol_np.astype(np.float32)
            vol_all.append(np.expand_dims(vol_np, 0))
        
        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=0).transpose([1,0,2,3])
            
        yield volume, dn


class unidirectional_model(segmentation_model):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs.keys())
        super().__init__(*args, **kwargs)
        self.debug_unidirectional = True


if __name__=='__main__':
    args = parse()
    
    # Load args.
    print("Loading experiment arguments.")
    args_path = os.path.join(args.experiment_path, 'args.txt')
    if not os.path.exists(args_path):
        raise ValueError(
            f'No `args.txt` found in experiment path: {args.experiment_path}'
        )
    model_parser = get_model_parser()
    with open(args_path, 'r') as f:
        saved_args = f.read().split('\n')[1:]
        saved_args[saved_args.index('--path')+1] = args.experiment_path
        if args.model_kwargs is not None:
            saved_args[saved_args.index('--model_kwargs')+1] = args.model_kwargs
        model_args = model_parser.parse_args(args=saved_args)
    
    print("Loading checkpoint at best epoch by validation score.")
    experiment_state = experiment(model_args)
    experiment_state.load_best_state()
    model = experiment_state.model['G']
    
    # Monkeypatch model to be unidirectional:
    # - Avoid sampling a code of a fixed size (may be incompatible with image
    #   size.
    # - Avoid some unnecessary computation for inference.
    model.debug_unidirectional = True
    model._forward.debug_unidirectional = True
    model._loss_D.debug_unidirectional = True
    model._loss_G.debug_unidirectional = True
    
    # Monkeypatch MI network out.
    model.separate_networks['mi_estimator'] = None
    model._loss_D.net['mi'] = None
    model._loss_G.net['mi'] = None
    
    print("Running model inference on test data.")
    if not os.path.exists(args.save_results_to):
        os.makedirs(args.save_results_to)
    segmentation = None
    for i, (vol, dn) in enumerate(data_loader(args.data_dir)):
        vol = preprocess(vol)
        for idx in range(0, len(vol), args.batch_size):
            A = torch.from_numpy(vol[idx:idx+args.batch_size]).float().cuda()
            B = torch.zeros(A.shape, dtype=torch.float, device=A.device)
            M = [np.zeros((1,)+A.shape[2:], dtype=np.int64)
                 for _ in range(len(A))]
            outputs = model(A, B, M)
            if segmentation is None:
                seg_shape = (vol.shape[0], 1, vol.shape[2], vol.shape[3])
                segmentation = np.zeros(seg_shape, dtype=np.uint8)
            x_AM = outputs['x_AM'].detach().cpu().numpy()
            x_AM_bin = x_AM > 0.5
            segmentation[idx:idx+args.batch_size] = x_AM_bin
        sitk_seg = sitk.GetImageFromArray(segmentation[:,0])
        sitk.WriteImage(sitk_seg, os.path.join(args.save_results_to,
                                               f'{dn}.nii.gz'))
        print(f'Processed {i+1}: {dn}')
