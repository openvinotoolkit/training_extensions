# Face Recognition

## Introduction
This repository contains training and inference code for face recognition neural networks. The face recognition network is based on the [MobileFaceNet](https://arxiv.org/abs/1804.07573) architecture equipped with [Squeeze-and-Excitation blocks](https://arxiv.org/abs/1709.01507). Landmark regression network is a simple CNN consisting from several stacked convolution and pooling layers. The code supports conversion to the ONNX\* format.

| Model Name | LFW accuracy | GFlops | MParams | Links |
| ---        | ---          | ---    | ---     | ---   |
| face-reidentification-retail-0095 | 0.9947 | 0.588 | 1.107 | [shapshot](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_se_focal_121000.pt), [configuration file](configs/face-reidentification-retail-0095.yaml) |
| MibileFaceNetSE_2x | 0.9942 | 1.155 | 2.197 | [shapshot](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_2x_se_121000.pt), [configuration file](configs/mobile_face_net_se_2x_vgg2.yaml) |

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5.2
* PyTorch\* 1.3 or higher
* OpenVINO™ 2020 R3 (or newer) with Python API

### Installation

To create and activate virtual Python environment follow [installation instructions](../README.md)


## Data preparation

1. For a face-recognition training, download the VGGFace2 data. We will refer to this folder as `$VGGFace2_ROOT`. Then align and crop this dataset using the provided annotation and script:
```bash
mv $VGGFace2_ROOT/test/* $VGGFace2_ROOT/train/

cat $VGGFace2_ROOT/meta/train_list.txt $VGGFace2_ROOT/meta/test_list.txt >> $VGGFace2_ROOT/meta/all_list.txt

cat $VGGFace2_ROOT/bb_landmark/loose_landmark_train.csv $VGGFace2_ROOT/bb_landmark/loose_landmark_test.csv >> $VGGFace2_ROOT/bb_landmark/loose_landmark_all.csv

cat $VGGFace2_ROOT/bb_landmark/loose_bb_train.csv $VGGFace2_ROOT/bb_landmark/loose_bb_test.csv >> $VGGFace2_ROOT/bb_landmark/loose_bb_all.csv

python3 ../../../external/deep-object-reid/convert_vgg_face_2.py
                      --data-root  $VGGFace2_ROOT
                      --output-dir $root/vggface2
```

2. For a face-recognition evaluation, download the [LFW](http://vis-www.cs.umass.edu/lfw/) data and [LFW landmarks](https://github.com/clcarwin/sphereface_pytorch/blob/master/data/lfw_landmark.txt).  Place everything in one folder, which will refer to as `$LFW_ROOT`.

The overall directory structure should look as follows:

```
root
├── vggface2
│   ├── all_list.txt
│   ├── n0000002
│   └── ....
│
└── lfw
    ├── lfw_landmark.txt
    ├── pairs_bench_crossval.txt
    ├── ...
```

### Configuration Files

The script for training and inference uses a configuration file
[default_config.py](https://github.com/openvinotoolkit/deep-object-reid/tree/ote/scripts/default_config.py), which consists of default parameters.
This file also has description of parameters.
Parameters that you wish to change must be in your own configuration file.
Example: [face-reidentification-retail-0095.yaml](configs/face-reidentification-retail-0095.yaml)

## Training

To start training, create or choose a configuration file and use the [main.py](https://github.com/openvinotoolkit/deep-object-reid/tree/ote/tools/main.py) script.

Example:

```bash
python ../../../external/deep-object-reid/tools/main.py \
    --root /path/to/datasets/directory/root \
    --config configs/face-reidentification-retail-0095.yaml
```

## Test
To test your network on the LFW dataset, set in a configuration file the `test.evaluate` parameter to `True`
and run a command like the one used for training.

## Training and evaluation of landmarks regression model

To train the landmarks regression model we need to use the original
unaligned VGGFace2 dataset:

```bash
python ../../../external/deep-object-reid/projects/landmarks_regression/train.py \
                --train_data_root $VGGFace2_ROOT/train/ \
                --train_list $VGGFace2_ROOT/meta/train_list.txt \
                --train_landmarks $VGGFace2_ROOT/bb_landmark/ \
                --dataset vgg --snap_folder <snapshots_folder>
```

To evaluate the trained model run the corresponding script:
```bash
python3 evaluate.py \
          --dataset vgg \
          --val_data_root $VGGFace2_ROOT/train/ \
          --val_list /mnt/big_ssd/VGGFace2/meta/test_list.txt \
          --val_landmarks /mnt/big_ssd/VGGFace2/bb_landmark/ \
          --snapshot <path to snashot>
```

**Note:** VGGFace2 contains auto-generated annotation of facial landmarks, therefore, this annotation is not very precise, but it's enuogh to train a decent model.

## Convert a PyTorch Model to the OpenVINO™ Format

To convert the obtained face recognition model, the following:

```bash
python ../../../external/deep-object-reid/tools/convert_to_onnx.py \
    --config /path/to/config/file.yaml \
    --output-name /path/to/output/model \
    --verbose
```

Name of the output model ends with `.onnx` automatically.
By default, the output model path is `model.onnx`. Be careful about the `load_weights` parameter
 in the configurations file. The `verbose` argument is non-required and switches on detailed output in conversion function.

To convert the trained landmark regression model launch the script:

```bash
python3 convert_onnx.py --snap <path to snapshot> --output_dir <output directory>
```
To convert the obtained ONNX format to OpenVINO™ IR, the following steps should be done:
* Make sure that OpenVINO environment is [initialized](../README.md).
* Launch the script from Model Optimizer directory:
```bash
python3 mo_onnx.py --input_model <path to obtained onnx> --output_dir <path to output dir> --reverse_input_channels --input_shape [1,3,128,128] --scale 255
``
