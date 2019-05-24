# Super Resolution Training Toolbox Pytorch
This code is intended for training Super Resolution (SR) algorithms in Pytorch. 

# Models
Two typologies are available for training at this point:

1. Single image super resolution network based on SRResNet architecture
(["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network"](https://arxiv.org/pdf/1609.04802.pdf)) but with reduced number of channels and depthwise convolution in decoder.
2. Attention-Based single image super resolution network (https://arxiv.org/pdf/1807.06779.pdf) with reduced number of channels and changes in network architecture.

# Results

| Model    | Set5, PSNRx3, dB | Set5, PSNRx4, dB |
| :------- | ----: | :---: |
| SmallModel    | 33.19 | 31.29 |

# Dependencies
pytorch 0.4+, python 3.5, opencv, skimage 0.14.1, numpy

# Training

*main.py* script should be used to start training process:

```
usage: main.py [-h] [--scale SCALE] [--model {SRResNetLight,SmallModel}]
               [--patch_size PATCH_SIZE [PATCH_SIZE ...]] [--border BORDER]
               [--aug_resize_factor_range AUG_RESIZE_FACTOR_RANGE [AUG_RESIZE_FACTOR_RANGE ...]]
               [--num_of_train_images NUM_OF_TRAIN_IMAGES]
               [--num_of_patches_per_image NUM_OF_PATCHES_PER_IMAGE]
               [--num_of_val_images NUM_OF_VAL_IMAGES] [--resume]
               [--batch_size BATCH_SIZE] [--num_of_epochs NUM_OF_EPOCHS]
               [--num_of_data_loader_threads NUM_OF_DATA_LOADER_THREADS]
               [--train_path TRAIN_PATH] [--validation_path VALIDATION_PATH]
               [--exp_name EXP_NAME] [--models_path MODELS_PATH] [--seed SEED]
               [--milestones MILESTONES [MILESTONES ...]]

Super Resolution PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --scale SCALE         Upsampling factor for SR
  --model {SRResNetLight,SmallModel}
                        SR model
  --patch_size PATCH_SIZE [PATCH_SIZE ...]
                        Patch size used for training (None - whole image)
  --border BORDER       Ignored border
  --aug_resize_factor_range AUG_RESIZE_FACTOR_RANGE [AUG_RESIZE_FACTOR_RANGE ...]
                        Range of resize factor for training patch, used for
                        augmentation
  --num_of_train_images NUM_OF_TRAIN_IMAGES
                        Number of training images (None - use all images)
  --num_of_patches_per_image NUM_OF_PATCHES_PER_IMAGE
                        Number of patches from one image
  --num_of_val_images NUM_OF_VAL_IMAGES
                        Number of val images (None - use all images)
  --resume              Resume training from the latest state
  --batch_size BATCH_SIZE
                        Training batch size
  --num_of_epochs NUM_OF_EPOCHS
                        Number of epochs to train for
  --num_of_data_loader_threads NUM_OF_DATA_LOADER_THREADS
                        Number of threads for data loader to use, Default: 1
  --train_path TRAIN_PATH
                        Path to train data
  --validation_path VALIDATION_PATH
                        Path to folder with val images
  --exp_name EXP_NAME   Experiment name
  --models_path MODELS_PATH
                        Path to models folder
  --seed SEED           Seed for random generators
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices, where learning rate decay is
                        applied
```

Example:
```
python ./main.py --batch_size 256 --num_of_epochs 100 --num_of_data_loader_threads 8 --train_path PATH_TO_TRAIN_DATA --validation_path PATH_TO_VAL_DATA --exp_name test --models_path PATH_TO_MODELS_PATH  --milestones 8 12 16 --scale 4 --patch_size 192 192 --model SRResNetLight --aug_resize_factor_range 0.8 1.2
```

# Testing

*test.py* script can be used to evaluate the trained model.

```
usage: test.py [-h] [--test_data_path TEST_DATA_PATH] [--exp_name EXP_NAME]
               [--models_path MODELS_PATH] [--scale SCALE] [--border BORDER]

PyTorch SR test

optional arguments:
  -h, --help            show this help message and exit
  --test_data_path TEST_DATA_PATH
                        path to test data
  --exp_name EXP_NAME   experiment name
  --models_path MODELS_PATH
                        path to models folder
  --scale SCALE         Upsampling factor for SR
  --border BORDER       Ignored border

```
Example:
```
python test.py --test_data_path PATH_TO_TEST_DATA --exp_name test --models_path PATH_TO_MODELS_PATH --scale 4 --border 4
```

# Checkpoints

Checkpoints can be downloaded [here](). Place *011_repro* (x4) and *021_repro2* (x3) into the model directory. Start training from the checkpoint with --resume flag.