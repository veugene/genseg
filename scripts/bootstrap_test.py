import argparse
import imp
import json
import os

from natsort import natsorted
import numpy as np
from scipy.stats import bootstrap
import torch
from tqdm import tqdm

from data_tools.io import data_flow


def parse():
    parser = argparse.ArgumentParser(description='Test whether a global Dice '
        'difference is significant with bootstrapping.')
    parser.add_argument('--experiment_paths_A',
                        nargs='+', type=str, required=True)
    parser.add_argument('--experiment_paths_B',
                        nargs='+', type=str, required=True)
    parser.add_argument('--confidence_level',
                        nargs='+', type=float, default=[0.999, 0.99, 0.95])
    return parser.parse_args()


DATASET_ARGS = [
    # All
    'data',
    'data_seed',
    
    # Brain
    'dataset',
    
    # Liver
    'data_fold',
    'data_split_seed',
    
    # Synthetic
    'n_clutter',
    'size_clutter',
    'size_output',
    'background_noise',
    'n_valid',
    #'unlabeled_digits',
    #'epoch_length',
]


def parse_experiment_args(experiment_path):
    if not os.path.exists(os.path.join(experiment_path, "args.txt")):
        raise ValueError(f'No args.txt in "{experiment_path}"')
    with open(os.path.join(experiment_path, "args.txt"), 'r') as f:
        # All arguments start with '--'.
        arg_tuple_list = f.read().split('\n--')[1:]
    args = {}
    for arg_tuple in arg_tuple_list:
        if '\n' not in arg_tuple:
            continue
        key, val = arg_tuple.split('\n', maxsplit=1)
        args[key] = val
    return args


def get_dataset_args(experiment_path):
    dataset_args = {}
    args = parse_experiment_args(experiment_path)
    for key in DATASET_ARGS:
        if key in args:
            dataset_args[key] = args[key]
    return dataset_args


def get_brain_dataset(dataset_args):
    raise NotImplementedError
    #import h5py
    #assert dataset_args['dataset'] == 'brats17'
    #rnd_state = np.random.RandomState(0)
    #hgg_indices = np.arange(0, 210)
    #lgg_indices = np.arange(0, 75)
    #rnd_state.shuffle(hgg_indices)
    #rnd_state.shuffle(lgg_indices)
    #hgg_test = hgg_indices[21:42]
    #lgg_test = lgg_indices[7:15]
    #hgg_file = h5py.File(
        #os.path.join(dataset_args['data'], 'hgg.h5'), mode='r')
    #lgg_file = h5py.File(
        #os.path.join(dataset_args['data'], 'lgg.h5'), mode='r')
    #dataset = []
    #for idx, case_id in enumerate(hgg_file.keys()):
        #if idx in hgg_test:
            #dataset.append(hgg_file[case_id])
    #for idx, case_id in enumerate(lgg_file.keys()):
        #if idx in lgg_test:
            #dataset.append(lgg_file[case_id])
    #return dataset


def get_liver_dataset(dataset_args):
    raise NotImplementedError


def mnist_preprocessor(batch):
    h, s, m, _ = zip(*batch[0])
    h = np.expand_dims(h, 1)
    s = np.expand_dims(s, 1)
    m = [np.expand_dims(x, 0) if x is not None else None for x in m]
    return h, s, m


def get_synthetic_dataset(dataset_args):
    from utils.data.cluttered_mnist import (
        setup_mnist_data,
        mnist_data_test,
    )
    data = setup_mnist_data(
        data_dir=dataset_args['data'],
        n_valid=int(dataset_args['n_valid']),
        n_clutter=int(dataset_args['n_clutter']),
        size_clutter=int(dataset_args['size_clutter']),
        size_output=int(dataset_args['size_output']),
        #unlabeled_digits=dataset_args['unlabeled_digits'],
        #gen_train_online=dataset_args['epoch_length'] is not None,
        background_noise=float(dataset_args['background_noise']),
        verbose=True,
        rng=np.random.RandomState(int(dataset_args['data_seed']))
    )
    dataset = mnist_data_test(data)
    loader = data_flow(
        [dataset],
        batch_size=200,
        preprocessor=mnist_preprocessor,
    )
    return loader


def get_dataloader(dataset_args):
    if 'dataset' in dataset_args:
        return get_brain_dataset(dataset_args)
    elif 'data_fold' in dataset_args:
        return get_liver_dataset(dataset_args)
    else:
        return get_synthetic_dataset(dataset_args)


def get_dataloader_and_assert_same_dataset(path_list):
    # Get the dataset arguments.
    dataset_args = get_dataset_args(path_list[0])
    
    # Assert that all experiments used the same dataset.
    for path in path_list[1:]:
        assert get_dataset_args(path) == dataset_args
    
    # Set up the dataset.
    dataloader = get_dataloader(dataset_args)
    return dataloader


def load_model(experiment_path):
    # Load checkpoint.
    path_list = natsorted(os.listdir(experiment_path))
    checkpoint_path = None
    for path in path_list[::-1]:
        if path.startswith('best_state_dict'):
            checkpoint_path = os.path.join(experiment_path, path)
            break
    assert checkpoint_path is not None
    saved_dict = torch.load(checkpoint_path)
    
    # Create the model.
    args = parse_experiment_args(experiment_path)
    model_kwargs = {}
    if 'model_kwargs' in args:
        model_kwargs = json.loads(args['model_kwargs'])
    module = imp.new_module('module')
    exec(saved_dict['model_as_str'], module.__dict__)
    model = getattr(module, 'build_model')(**model_kwargs)
    
    # Load weights into model and move to GPU.
    for key in model.keys():
        model[key].load_state_dict(saved_dict[key]['model_state'], strict=False)
        model[key].cuda()
        model[key].eval()
    
    return model


def dice(counts):
    n, d = zip(*counts)
    score = (2 * np.sum(n) + 1) / (np.sum(d) + 1)
    return score


def mean_dice_diff(results_A, results_B, axis):
    A = results_A.sum(-1)   # seeds, 2, n_resamples
    B = results_B.sum(-1)   # seeds, 2, n_resamples
    dice_A = (2 * A[:, 0] + 1) / (A[:, 1] + 1)  # seeds, n_resamples
    dice_B = (2 * B[:, 0] + 1) / (B[:, 1] + 1)  # seeds, n_resamples
    diff = dice_A.mean(0) - dice_B.mean(0)  # n_resamples
    return diff


if __name__ == '__main__':
    args = parse()
    
    # Set up data.
    dataloader = get_dataloader_and_assert_same_dataset(
        args.experiment_paths_A + args.experiment_paths_B)
    
    # Predict on data.
    path_list = {
        'A': args.experiment_paths_A,
        'B': args.experiment_paths_B,
    }
    results = {'A': [], 'B': []}
    print('Predicting...')
    for key in ('A', 'B'):
        for experiment_path in path_list[key]:
            print(f'{key}: {experiment_path}')
            model = load_model(experiment_path)
            counts = []
            pbar = tqdm(dataloader.flow(), total=len(dataloader))
            for h, s, m in pbar:
                h = torch.from_numpy(h).cuda()
                s = torch.from_numpy(s).cuda()
                p = model['G'](s, h, m)['x_AM'].detach().cpu().numpy()
                for idx in range(len(p)):
                    counts.append(
                        (
                            np.sum(m[idx].flatten() * p[idx].flatten()),
                            np.sum(m[idx]) + np.sum(p[idx]),
                        )
                    )
                pbar.set_postfix({'dice': dice(counts)})
            results[key].append(counts)
            print(f'Dice = {dice(counts):.4f}')
            del model
    
    def is_significant(confidence_level, paired):
        result = bootstrap(
            data=[np.stack(results['A']), np.stack(results['B'])],
            statistic=mean_dice_diff,
            method='basic',
            paired=paired,
            axis=1,
            confidence_level=confidence_level,
        )
        
        # Significant difference if CI does not contain zero.
        low = result.confidence_interval.low
        high = result.confidence_interval.high
        if low <= 0 and high >= 0:
            return False
        return True
    
    # Bootstrap
    print('Bootstrapping...')
    for confidence_level in args.confidence_level:
        for is_paired in [True, False]:
            if is_paired:
                is_paired_str = 'paired'
            else:
                is_paired_str = 'unpaired'
            print(
                f'Confidence level {confidence_level} '
                f'({is_paired_str}) : '
                f'{is_significant(confidence_level, is_paired)}'
            )