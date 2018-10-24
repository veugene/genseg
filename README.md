# Bidomain segmentation

## Initialization

Run `git submodule init` to download submodules.  
Run `source link_submodules.sh` from *within the root directory of the source tree* to set up submodule links and add then to the PYTHONPATH.

Requires:
- tqdm
- SimpleITK

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
`cluttered_mnist_segmentation.py` : Cluttered MNIST

Task launchers are used to start/resume an experiment.

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
CUDA_VISIBLE_DEVICES="0" python3 brats_segmentation.py` --name "bds3_003_xid50_cyc50 (f0.01, D_lr 0.001) [b0.3]" --model_from model/configs/brats_2017/bds3/bds3_003_xid50_cyc50.py --batch_size_train 20 --batch_size_valid 20 --epochs 1000000 --rseed 1234 --optimizer '{"G": "amsgrad", "D": "amsgrad"}' --opt_kwargs '{"G": {"betas": [0.5, 0.999], "lr": 0.001}, "D": {"betas": [0.5, 0.999], "lr": 0.01}}' --save_path experiments/brats_2017/bds3 --n_vis 8 --weight_decay 0.0001 --dataset brats17 --orientation 1 --data_dir=./data/brats/2017/hemispheres_b0.3_t0.01/ --labeled_fraction 0.01 --augment_data --nb_proc_workers 2
```

Optimizer arguments are passed as a JSON string through the `--opt_kwargs` argument.

#### Example: resuming a brats experiment

Resuming the above experiment could be done with:
```
CUDA_VISIBLE_DEVICES="0" python3 brats_segmentation.py --resume_from experiments/brats_2017/bds3/<experiment directory>
```

Upon resuming, the model configuration file is loaded from the saved checkpoint. All arguments passed upon initializing the experiment are loaded as well. **Any of these can be over-ridden by simply passing them again with the resuming command.**
