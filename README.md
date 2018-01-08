# CNN for Synovitis Segmentation in Rheumatoid Arthritis

## Overview

This is an implementation of a fully convolutional CNN for segmenting synovitis in wrist MRIs of patients with Rheumatoid Arthritis, in Tensorflow. The network is trained end-to-end on 512 x 512 x 20 MR volumes and outputs a voxel-wise segmentation of the input volume. It's trained using dice coefficient as a loss function and performs with an average dice coefficient of 0.61 on a test set of 10 MR volumes from UCSF patients. Check out the [paper](https://github.com/cosmicac/ucsf-mri-seg/blob/refactor/paper/raseg.pdf) for more details.

## Architecture

The architecture of the network is depicted below. 
![architecture](https://github.com/cosmicac/ucsf-mri-seg/blob/refactor/paper/figures/fig2.png "Network Architecture")

## Examples

Below are some examples of ground truth segmentations versus what the network predicts. 
![examples](https://github.com/cosmicac/ucsf-mri-seg/blob/refactor/paper/figures/fig3.png "Prediction Examples")

## Usage

Python 2.7 and Tensorflow 0.12 were used in this implementation. 

Place `images_and_labels.npy` and `pre_images.npy` in a `datasets` folder. These are available from here. These are the training volumes and labels arranged in a numpy array. 

### Make the binaries

Use `make_datasets.py` to flatten the numpy arrays into binaries. The binaries will be dumped into a `datasets/bins/` directory. The required `tag` argument is a string you create to help identify the particular dataset you are creating. The required `val` argument is the desired size of the validation set.

```bash
python make_datasets.py --tag fullconv_extended --val 10
```

### Train the model

Use `raseg_train.py` to train the model. This will train the model with dice coefficient loss using the Adam optimizer. Hyper-parameters are located in `raseg_model.py`. The `tag` argument is again required to identify the binaries to use. The model will be saved in a `models` directory.

```bash
python raseg_train.py --tag fullconv_extended
```

### Predict on a volume

Use `raseg_predict.py` to use a trained model to segment a particular volume. The `tag` argument is required. The `imgn` argument is the index of the volume you want to predict in the `images_and_labels.npy` numpy array. This will save a numpy array of the model's predictions in an `preds` folder, and save pictures of the ground truth segmentations versus the model's outputs in a `pictures` folder.

```bash
python raseg_predict.py --tag fullconv_extended --imgn 51
```
