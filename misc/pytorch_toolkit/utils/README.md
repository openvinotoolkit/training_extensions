# Script to Convert Pretrained Models from PyTorch* to ONNX*

This repository contains the script for conversion of public models pretrained in
PyTorch to the ONNX format. The script enables conversion of models from
torchvision and for public models, description and pretrained weights of which are
available to clone or download from the internet resources.

## Supported Models

The models supported in the current version of the script:

* torchvision models:
    * ResNet-50-v1
    * Inception-v3

* Public pretrained models
    * MobileNetV2 (<https://github.com/tonylins/pytorch-mobilenet-v2>)

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5.2
* PyTorch\* 1.0.1
* torchvision\* 0.2.2 (for torchvision models support)
* ONNX\* 1.4.1

### Installation

> **TIP:** Use this script under the Python virtual environment to avoid possible conflicts between
> already installed Python packages and required packages for the script.

To use the virtual environment, create and activate it:

```bash
python3 -m virtualenv -p `which python3` <directory_for_environment>
source <directory_for_environment>/bin/activate
```
Install requirements:

```bash
pip3 install -r requirements.txt
```
Now you can work with the script.

To deactivate the virtual environment, use the following command:

```bash
deactivate
```

## Usage

The script takes the following input arguments:

* `--model-name` - name of the PyTorch model to convert. Currently available model names are:
    * resnet-v1-50
    * inception-v3
    * mobilenet-v2
* `--weights` - path to a `.pth` or `.pth.tar` file with downloaded pretrained PyTorch weights
* `--input-shape` - input blob shape, given by four space-separated positive integer values for `batch size`,
  `number of channels`, `height`, and `width` in the order defined for the chosen model
* `--output-file` - path to the output `.onnx` file with the converted model

Optional arguments:

* `--model-path` - path to a directory with Python file(s), containing description of a PyTorch model chosen for
  conversion. This parameter should be provided for public models that are not a part of torchvision package.
* `--input-names` - space-separated (if several) names of input layers. The input layers names are presented by
  these values in the ONNX model. Indexes of layers are used if this argument was not provided.
* `--output-names` - space-separated (if several) names of output layers. The output layers names are presented by
  these values in the ONNX model. Indexes of layers are used if this argument was not provided.

Refer to `-h, --help` option to get the full list of script arguments.

## Example

To convert the ResNet-50-v1 model from torchvision, use the following command:

```bash
python3 pytorch_to_onnx.py \
    --model-name resnet-v1-50 \
    --weights <path_to_downloaded_pretrained_weights>/resnet50-19c8e357.pth \
    --input-shape 1 3 224 224 \
    --output-file <path_to_save_converted_model>/resnet-v1-50.onnx \
    --input-names data \
    --output-names prob
```
