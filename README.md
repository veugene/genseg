# Bidomain segmentation

## Initialization

Run `pip install -r requirements.txt` to install dependencies.__
Run `git submodule init` to initialize submodules.  
Run `git submodule update` to download submodules.  
Run `source link_submodules.sh` from *within the root directory of the source tree* to set up submodule links and add then to the PYTHONPATH.  

## Models

`models/bd_segmentation.py` : model 3  
`models/bd_segmentation_residual.py` : model 2  
`models/ae_segmentation.py` : autoencoding+segmentation  

A model is a pytorch module that determines compute and update rules. It takes sub-network definitions as initialization arguments (eg. encoder, decoders, discriminators, etc.).

A model is instantiated in a *configuration file* with a `build_model()` function. Any other code/functions could exist in the file to help with building all sub-networks, etc. The `build_model()` function must return one of the following:  
1. An instantiated model.  
2. A dictionary containing the instantiated model and any other models.  

Option (2) is useful when the model contains subnetworks that are trained by separate optimizers. In that case, an optimizer is created for every item in the dictionary. Separate optimization methods and optimizer parameters could be passed to each optimizer via the arguments to the task launchers.

Example of option (2), where discriminators are trained separately:
```python
def build_model():
    # Code setting up model.
    return {'G' : model,
            'D' : nn.ModuleList([model.separate_networks['disc_A'],
                                 model.separate_networks['disc_B']])}
```

## Tasks

`brats_segmentation.py` : BRATS  
`lits_segmentation.py` : LITS__
`cluttered_mnist_segmentation.py` : Cluttered MNIST  

Task launchers are used to start/resume an experiment.

### BRATS

This is the 2017 version of the BRATS (brain tumour segmentation) data from https://www.med.upenn.edu/sbia/brats2017/data.html. Once downloaded to `<download_dir>`, this data can be prepared for training using a provided script, as follows:
```
python scripts/data_preparation/prepare_brats_data_hemispheres.py --data_dir <download_dir>/HGG --save_to data/brats_2017_b0.25_t0.01/hgg.h5 --min_tumor_fraction 0.01 --min_brain_fraction 0.25
```
```
python scripts/data_preparation/prepare_brats_data_hemispheres.py --data_dir <download_dir>/LGG --save_to data/brats_2017_b0.25_t0.01/lgg.h5 --min_tumor_fraction 0.01 --min_brain_fraction 0.25
```
Data preparation creates a new dataset based on BRATS that contains 2D hemispheres, split into sick and healthy subsets.

### LiTS

This is the 2017 Liver Tumour Segmentation (LiTS) challenge data from the https://competitions.codalab.org/competitions/17094. After downloading this data to `<download_dir>/Training_Batch1.zip` and `<download_dir>/Training_Batch2.zip`, it must be unzipped to some `<data_dir>` and the `.nii` NIFTI files must be gunzip compressed to `.nii.gz` files, as follows (in BASH):
```
for fn in <download_dir>/Training_Batch{1,2}.zip; do unzip -j $fn -d <data_dir>; done
for fn in <data_dir>/*.nii; do echo "Compressing $fn" && gunzip $fn; done
```
This data can then be prepared for training using as a provided script, as follows:
```
python scripts/data_preparation/prepare_lits.py <data_dir> --path_create data/lits/lits.h5
```

### MNIST

The cluttered MNIST digit data is created automatically by the MNIST task launcher from MNIST data that is also downloaded automatically.

### Launching

#### Example: launching a BRATS experiment

In this example, the following model configuration is used:
`model/configs/brats_2017/bds3/bds3_003_xid50_cyc50.py`

In this specific configuration file, the decoder is mostly shared for the segmentation path. When run in `mode=0` (passed in `forward` call; default), the decoder outputs an image, with `tanh` normalization at the output; when run in `mode=1`, it outputs a segmentation mask, with `sigmoid` normalization at the output. Modes 0 and 1 differ in three ways:
1. The final norm/nonlinearity/convolution block is unique for each mode.
2. The final nonlinearity is `tanh` in mode 0 and `sigmoid` in mode 1.
3. Every block in the decoder is normalized with `layer_normalization` in mode 0 and with adaptive instance normalization in mode 1.
Adaptive instance normalization uses parameters predicted by an MLP when the decoder is run in mode 0. They are passed to the mode 1 decoder as `skip_info`.

An example of an experiment launched with this config is:
```
python brats_segmentation.py --path "experiments/brats_2017/bds3/bds3_003_xid50_cyc50 (f0.01, D_lr 0.001) [b0.3]" --model_from model/configs/brats_2017/bds3/bds3_003_xid50_cyc50.py --batch_size_train 20 --batch_size_valid 20 --epochs 1000000 --rseed 1234 --optimizer '{"G": "amsgrad", "D": "amsgrad"}' --opt_kwargs '{"G": {"betas": [0.5, 0.999], "lr": 0.001}, "D": {"betas": [0.5, 0.999], "lr": 0.01}}' --n_vis 8 --weight_decay 0.0001 --dataset brats17 --orientation 1 --data_dir=./data/brats/2017/hemispheres_b0.3_t0.01/ --labeled_fraction 0.01 --augment_data --nb_proc_workers 2
```

Optimizer arguments are passed as a JSON string through the `--opt_kwargs` argument.

Note that if `CUDA_VISIBLE_DEVICES` is not set to specify which GPUs to use, the model will attempt to use all available GPUs. The code is multi-GPU capable but no serious training has been done on multiple GPUs. Use multiple GPUs with caution. There have been bugs in pytorch (hopefully fixed now) that made it either cause some layers to fail to be updated or fail to be resumed.

#### Example: resuming a brats experiment

Resuming the above experiment could be done with:
```
python brats_segmentation.py --path "experiments/brats_2017/bds3/bds3_003_xid50_cyc50 (f0.01, D_lr 0.001) [b0.3]"
```

Upon resuming, the model configuration file is loaded from the saved checkpoint. All arguments passed upon initializing the experiment are loaded as well. **Any of these can be over-ridden by simply passing them again with the resuming command.**

## Dispatching on a compute cluster

To launch an experiment on a cluster, simply run the launcher with the appropriate `dispatch` argument and the task launcher will set up and queue the job on the cluster. Each cluster has cluster-specific arguments that can be set (see `--help`).

### Compute Canada ###

Experiments should be launched from one of the login nodes of a compute canada cluster. The launcher then sets up and queues the job on the cluster.

Note: SLURM setup for compute canada could be easily extended to other SLURM based clusters.

To the task arguments, add the argument `--dispatch_dgx`, along with any aditional DGX-specific arguments:

`--account` : the compute canada account to use for requesting resources  
`--cca_gpu` : number of GPUs to request  
`--cca_cpu` : number of CPU cores to request  
`--cca_mem` : amount of memory to request, as a string (eg. '12G')  
`--time` : the amount of time to request the job for (see `sbatch` time syntax)  

When dispatching on a compute canada cluster, a daemon is created that requeues any jobs that time out, allowing them to resume. This allows requesting a short run time which makes it much more likely to get high priority resources; an optimal run time request is for 3h (`--time "3:0:0"`).
