# Instance Segmentation in PyTorch

This repository contains inference and training code for Mask R-CNN like
networks. Models code is designed to enable ONNX export (with custom operations)
and inference on CPU via OpenVINO.
[Detectron](https://github.com/facebookresearch/Detectron) and
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
models are wrapped to export their weights to ONNX and OpenVINO.

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* PyTorch 0.4.1
* OpenVINO 2019 R1 with Python API

### Installation

To install required dependencies run

```bash
$ cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

To install the package itself run

```bash
$ pip3 install -e .
```

### Get pretrained models

A bunch of top-performing models from Detectron and maskrcnn-benchmark projects
can be easily obtained and automatically prepared for running in PyTorch and
OpenVINO, via `tools/download_pretrained_weights.py` script. By default script
requires no parameters to run, for more details on its configuration run it with
a `-h` key. This script could also be considered as a sample, showing how models
are supposed to be converted to ONNX/IR.

### Download MS COCO dataset

To be able to train networks and/or get quality metrics for pre-trained ones,
one have to download MS COCO dataset. Download [the
dataset](http://cocodataset.org/#download) (train, val and annotations) and
unpack it to `./data/coco` folder. The result structure of the folder should be:
```
data
└── coco
    ├── annotations
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    |
    └── images
        ├── train2017
        └── val2017
```


## Training

To train Mask R-CNN model run

```bash
$ python3 tools/train.py
```

This script has a default configuration which conforms to end-to-end Mask R-CNN
baseline from Detectron.
To view all arguments available to configure run

```bash
$ python3 tools/train.py --help
```

For fine-tuning just pass a path to a model's weights file to the training script.

Another option for training models is to use dedicated scripts for particular
models (like `tools/train_0050.py`) that solely encapsulate the training setup.

Instructions for training models from OpenVINO Open Model Zoo
can be found in [OpenVINO.md](OpenVINO.md)

## Demo

`tools/demo.py` script implements a live demo application, that runs a given
Mask R-CNN like model on a set of images or a video and shows resulting instance
segmentation mask. Both PyTorch and OpenVINO backends are supported.

### PyTorch

As input, the demo application takes:

* a model
  * a path to PyTorch `nn.Module` implementing the network of interest specified
    with a command line argument `--model`
  * a path to a file with pre-trained weights specified with a command line
    argument `--ckpt`
* a source of data:
  * a path to a single image file or a folder with image files specified with
    a command line argument `--images`
  * a path to a video file or a numeric ID of a web-camera specified with
    a command line argument `--video`
* preprocessing parameters
  * an image resize mode and target resolution. Two options are available here.
    `--fix_max SCALE MAXSIZE` command line argument forces image to be resized
    to such a size, that its shorter and larger sides are not greater than
    `SCALE` and `MAXSIZE` respectively, while original aspect ratio is left
    unchanged. `--fit_window MAXHEIGHT MAXWIDTH` option enables the mode,
    when image height and width are made not greater that `MAXHEIGHT` and
    `MAXWIDTH` respectively, while original aspect ratio is preserved.
  * a mean value subtracted from every pixel of an image
    (`--mean_pixel` argument)
* extra options controlling visualization and performance statistics collection.
  Please refer to script's help message (run it with `-h` argument)
  for more details.

For example, assuming that `tools/download_pretrained_weights.py` script with
default options has been used to fetch public pre-trained models, to run demo on
live video stream from a webcam using ResNet50-FPN Mask R-CNN model for instance
segmentation run the following command:

```bash
$ python3 tools/demo.py \
    --dataset coco_2017_val \
    --ckpt data/pretrained_models/converted/coco/detectron/mask_rcnn_resnet50_fpn_2x.pth \
    --mean_pixel 102.9801 115.9465 122.7717 \
    --fit_window 800 1333 \
    --video 0 \
    --delay 1 \
    --show_fps \
    pytorch \
    --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN \
    --show_flops
```

> **Note:** Use `CUDA_VISIBLE_DEVICES` environment variable to configure which
(if any) GPUs to use during evaluation. If empty value is assigned, PyTorch uses
CPU backend.

### OpenVINO

The same demo application may be used for running instance segmentation models
on CPU via OpenVINO. Almost the same set of parameters is available in this
case. The major difference is that model (`--model` argument) should be defined
as a path to an XML file with OpenVINO IR description, rather that a PyTorch
class, and `--ckpt` argument should point to a BIN file with OpenVINO IR
weights.

Example:
```bash
$ python3 tools/demo.py \
    --dataset coco_2017_val \
    --ckpt data/pretrained_models/ir/coco/detectron/mask_rcnn_resnet50_fpn_2x.bin \
    --fit_window 800 1333 \
    --video 0 \
    --delay 1 \
    --show_fps \
    openvino \
    --model data/pretrained_models/ir/coco/detectron/mask_rcnn_resnet50_fpn_2x.xml
```

> **Note:** For most of the Detectron and maskrcnn-benchmark models mean pixel
  value of [102.9801, 115.9465, 122.7717] is used while running with a PyTorch
  backend. But at the same time this value is integrated into the model itself
  by OpenVINO Model Optimizer during export to IR, so there is no need
  to specify this value when running with an OpenVINO backend.

## Evaluation

`tools/test.py` script is designed for quality evaluation of instance
segmentation models. The script has almost the same interface as a demo script,
and supports both PyTorch and OpenVINO backends.

### PyTorch

For example, to evaluate ResNet50-FPN Mask R-CNN model on COCO 2017 Val dataset
using PyTorch backend run:

```bash
$ python3 tools/test.py \
    --dataset coco_2017_val \
    --ckpt data/pretrained_models/converted/coco/detectron/mask_rcnn_resnet50_fpn_2x.pth \
    --mean_pixel 102.9801 115.9465 122.7717 \
    --fit_max 800 1333 \
    pytorch \
    --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN \
    --show_flops
```

> **Note:** Use `CUDA_VISIBLE_DEVICES` environment variable to configure which
(if any) GPUs to use during evaluation. If empty value is assigned, PyTorch uses
CPU backend.

### OpenVINO

For example, to evaluate ResNet50-FPN Mask R-CNN model on COCO 2017 Val dataset
using OpenVINO backend run:

```bash
$ python3 tools/test.py \
    --dataset coco_2017_val \
    --ckpt data/pretrained_models/ir/coco/detectron/mask_rcnn_resnet50_fpn_2x.bin \
    --fit_window 800 1333 \
    openvino \
    --model data/pretrained_models/ir/coco/detectron/mask_rcnn_resnet50_fpn_2x.xml
```

> **Note:** Default quality evaluation protocol for Mask R-CNN model uses
`fit_max` image resize mode at a preprocessing stage (see the note above about
resize modes). Though, by default OpenVINO IR models created by
`tools/download_pretrained_weights.py` script are configured to work properly
with `fit_window` mode only. This has no difference for landscape-oriented
images, but affects portrait-oriented ones. So to directly reproduce reference
quality numbers one has to export PyTorch models to OpenVINO IR manually setting
`MAXSIZE`x`MAXSIZE` input resolution. This will be fixed in later releases.


## Export PyTorch models to OpenVINO

To run the model via OpenVINO one has to export PyTorch model to ONNX first and
then convert it to OpenVINO Internal Representation (IR) using Model Optimizer.

### Export to ONNX

`tools/convert_to_onnx.py` script exports a given model to ONNX representation.

As input, the script takes:

* a model
  * a path to PyTorch `nn.Module` implementing the network of interest specified
    with a command line argument `--model`
  * a path to a file with pre-trained weights specified with a command line
    argument `--ckpt`
* a number of classes that network detects specified either directly using
  a `-nc NUMBER_OF_CLASSES` argument or implicitly by specifying a dataset
  the network was trained or supposed to work on using
  a `--dataset DATASET_NAME` argument
* an output ONNX file path specified with a command line argument
  `--output_file`
* an input resolution the network should work on specified with a command line
  argument `--input_size` in a `HEIGHT WIDTH` format.

For example, here is the command used inside
`tools/download_pretrained_weights.py` script to export ResNet50-FPN Mask R-CNN
model to ONNX representation:

```bash
$ python3 tools/convert_to_onnx.py \
    --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNet50FPNMaskRCNN \
    --ckpt data/pretrained_models/converted/coco/detectron/mask_rcnn_resnet50_fpn_2x.pth \
    --input_size 800 1344 \
    --dataset coco_2017_val \
    --show_flops \
    --output_file data/pretrained_models/onnx/coco/detectron/mask_rcnn_resnet50_fpn_2x.onnx
```

> **Note:** Most of FPN Mask R-CNN models assumes to have an input with a size
divisible by 32. So even when image resize is configured to work in `fit_max`
mode with `SCALE` 800 and `MAXSIZE` 1333, maximal input resolution is actually
800x1344 to ensure divisibility.


### Convert to IR


Conversion from ONNX model representation to OpenVINO IR is straightforward and
handled by OpenVINO Model Optimizer. Please refer to Model Optimizer
documentation for details on how it works.

For example, here is the command used inside
`tools/download_pretrained_weights.py` script to export ResNet50-FPN Mask R-CNN
model to IR given its ONNX representation:

```bash
$ mo.py \
    --framework onnx \
    --input_model data/pretrained_models/onnx/coco/detectron/mask_rcnn_resnet50_fpn_2x.onnx \
    --output_dir data/pretrained_models/ir/coco/detectron/ \
    --input "im_data,im_info" \
    --output "boxes,scores,classes,batch_ids,raw_masks" \
    --mean_values "im_data[102.9801,115.9465,122.7717],im_info[0,0,0]"
```
