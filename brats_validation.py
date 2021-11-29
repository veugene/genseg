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
        "volume is one batch. No cropping to the brain is done."
        "\n"
        "Given multiple models, computes an ensemble result. For each model, "
        "additionally computes an ensemble over the 4 possible flips.")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('save_results_to', type=str)
    parser.add_argument('--experiment_path', type=str, nargs='+')
    parser.add_argument('--model_kwargs', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=80)
    return parser.parse_args()


AUGMENTATIONS = [
    lambda x: x,
    lambda x: np.flip(x, axis=2),
    lambda x: np.flip(x, axis=3),
    lambda x: np.flip(np.flip(x, axis=3), axis=2)
]


def preprocess(volume):
    volume_out = volume.copy()
    
    # Mean center and normalize by std.
    brain_mask = volume_out!=0
    volume_out[brain_mask] -= volume_out[brain_mask].mean()
    volume_out[brain_mask] /= volume_out[brain_mask].std()*5    # fit in tanh
    
    return volume_out


def data_loader(data_dir):
    tags = ['flair', 't1ce', 't1', 't2', 'seg']
    for dn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, dn)
        if not os.path.isdir(path):
            continue
        
        # Match filenames to tags.
        # NOTE: t1ce chosen when both t1 and t1ce matched
        fn_dict = OrderedDict()
        for fn in sorted(os.listdir(path)):
            match = [t for t in tags if t in fn]
            if len(match) == 0:
                continue
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
                segmentation = np.expand_dims(segmentation, 1)
            else:
                vol_np = vol_np.astype(np.float32)
                vol_all.append(np.expand_dims(vol_np, 1))
        
        # Concatenate on channel axis.
        volume = np.concatenate(vol_all, axis=1)
            
        yield volume, segmentation, dn


def load_model(experiment_path, model_kwargs):
    # Load args.
    print("Loading experiment arguments.")
    args_path = os.path.join(experiment_path, 'args.txt')
    if not os.path.exists(args_path):
        raise ValueError(
            f'No `args.txt` found in experiment path: {experiment_path}'
        )
    model_parser = get_model_parser()
    with open(args_path, 'r') as f:
        saved_args = f.read().split('\n')[1:]
        saved_args[saved_args.index('--path')+1] = experiment_path
        if model_kwargs is not None:
            saved_args[saved_args.index('--model_kwargs')+1] = model_kwargs
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
    
    return model


class unidirectional_model(segmentation_model):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs.keys())
        super().__init__(*args, **kwargs)
        self.debug_unidirectional = True


if __name__=='__main__':
    args = parse()
    models = []
    for path in args.experiment_path:
        m = load_model(path, args.model_kwargs)
        models.append(m)
    
    print("Running model inference on test data.")
    TP = FP = FN = 0
    dice = None
    if not os.path.exists(args.save_results_to):
        os.makedirs(args.save_results_to)
    for i, (vol, seg, dn) in enumerate(data_loader(args.data_dir)):
        vol = preprocess(vol)
        prediction = None
        for augment in AUGMENTATIONS:
            vol_input = augment(vol)
            for idx in range(0, len(vol), args.batch_size):
                A_np = vol_input[idx:idx+args.batch_size].copy()
                A = torch.from_numpy(A_np).float().cuda()
                B = torch.zeros(A.shape, dtype=torch.float, device=A.device)
                M = [np.zeros((1,)+A.shape[2:], dtype=np.int64)
                    for _ in range(len(A))]
                for m in models:
                    with torch.no_grad():
                        outputs = m(A, B, M)
                    if prediction is None:
                        pred_shape = (vol.shape[0], 1,
                                     vol.shape[2], vol.shape[3])
                        prediction = np.zeros(pred_shape, dtype=float)
                    x_AM = augment(outputs['x_AM'].detach().cpu().numpy())
                    x_AM_bin = x_AM > 0.5
                    prediction[idx:idx+args.batch_size] += x_AM_bin
        prediction /= len(AUGMENTATIONS)*len(models)
        prediction = prediction.astype(np.uint8)
        sitk_pred = sitk.GetImageFromArray(prediction[:,0])
        sitk.WriteImage(sitk_pred, os.path.join(args.save_results_to,
                                                f'{dn}.nii.gz'))
        
        # Compute values for Dice if there is a reference segmentation.
        if seg is not None:
            TP += np.count_nonzero(prediction[seg>0])
            FP += np.count_nonzero(prediction[seg==0])
            FN += np.count_nonzero(seg[prediction==0])
            dice = 2*TP/(2*TP+FP+FN)
        if dice is not None:
            print(f'Processed {i+1}: {dn} -- accumulated Dice so far is {dice}')
        else:
            print(f'Processed {i+1}: {dn}')
    if dice is not None:
        print(f'Dice score is {2*TP/(2*TP+FP+FN)}.')
