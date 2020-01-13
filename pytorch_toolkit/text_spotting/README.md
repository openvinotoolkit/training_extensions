# Text Spotting in PyTorch

This repository contains inference and training code for Text Spotting models based on Mask R-CNN like networks. Models code is designed to enable ONNX export (with custom operations) and inference on CPU via OpenVINO.

## Setup

### Prerequisites

* Ubuntu 16.04
* GCC 7.4.0
* Python 3.7.4
* PyTorch (custom, see Installation section)
* CUDA 10.1
* OpenVINO 2019 R4 with Python API

### Installation

0. Define your working directory.
```bash
export WORK_DIR=~/work_dir
mkdir -p $WORK_DIR
```

1. Create virtual environment and activate it.
```bash
virtualenv -p python3.7 --prompt="(text_spotting)" $WORK_DIR/venv
source $WORK_DIR/venv/bin/activate
```

2. Clone OpenVINO Training Extensions and install Text Spotting dependencies.
```bash
cd $WORK_DIR
git clone https://github.com/opencv/openvino_training_extensions.git
cd $WORK_DIR/openvino_training_extensions/pytorch_toolkit/text_spotting
cat requirements.txt | xargs -n 1 -L 1 pip3 install
```

3. Clone custom version of PyTorch to resolve known problems with ONNX export of custom python layers.
```bash
cd $WORK_DIR
git clone https://github.com/Ilya-Krylov/pytorch.git
cd $WORK_DIR/pytorch
git checkout -b enable_export_of_custom_onnx_operations_with_tuples_as_output origin/enable_export_of_custom_onnx_operations_with_tuples_as_output
git submodule update --init --recursive
python setup.py install
```

4. Clone custom version of torchvision and install it. You might need to updated your ffmpeg up to version 4.x and install libavcodec-dev.
```bash
cd $WORK_DIR
git clone https://github.com/pytorch/vision.git
cd $WORK_DIR/vision
git checkout be6dd4720652d630e95d968be2a4e1ae62f8807e
python setup.py install
```

5. Build Instance Segmentation (Segmentoly) package that is base for Text Spotting.
```bash
cd $WORK_DIR/openvino_training_extensions/pytorch_toolkit/instance_segmentation
python setup.py develop build_ext
```

6. Install Text Spotting.
```bash
cd $WORK_DIR/openvino_training_extensions/pytorch_toolkit/text_spotting
python setup.py develop
```

### Download dataset

To be able to train networks and/or get quality metrics for pre-trained ones,
one have to download one dataset at least.
* https://rrc.cvc.uab.es/ - ICDAR2013 (Focused Scene Text), ICDAR2013 (Incidental Scene Text), ICDAR2017 (MLT), ... .
* http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500) MSRA-TD500.
* https://bgshih.github.io/cocotext/ COCO-Text.
* ...

### Convert dataset

Extract downloaded datasets in following images_folder:  `$WORK_DIR/openvino_training_extensions/pytorch_toolkit/text_spotting/data/coco`

Convert extracted datasets to format that is used internally.

```bash
python3 tools/create_dataset.py --config dataset_train.json --output IC13TRAINTEST_IC15TRAIN_MSRATD500TRAINTEST_COCOTEXTTRAINVAL.json
python3 tools/create_dataset.py --config dataset_test.json --output IC15TEST.json
```


The dataset_\*.json should look like:

```
[
  {
    "name": "ICDAR2013DatasetConverter",
    "kwargs": {
      "images_folder": "icdar2013/Challenge2_Training_Task12_Images",
      "annotations_folder": "icdar2013/Challenge2_Training_Task1_GT",
      "is_train": true
    }
  },
  {
    "name": "ICDAR2013DatasetConverter",
    "kwargs": {
      "images_folder": "icdar2013/Challenge2_Test_Task12_Images",
      "annotations_folder": "icdar2013/Challenge2_Test_Task1_GT",
      "is_train": false
    }
  },
  {
    "name": "ICDAR2015DatasetConverter",
    "kwargs": {
      "images_folder": "icdar2015/ch4_training_images",
      "annotations_folder": "icdar2015/ch4_training_localization_transcription_gt",
      "is_train": true
    }
  }
]
```

Examples of dataset_configuration.json can be found in `openvino_training_extensions/pytorch_toolkit/text_spotting/datasets`.

## Training

To train Text Spotter model run (**do not forget to point where training and testing datasets are located inside text-spotting-0001.json**):

```bash
python3 tools/train.py configs/text-spotting-0001.json
```

One can point to pre-trained model (checkpoint) inside configuration file to start training from pre-trained weights. See `configs/text-spotting-0001.json`.
```
...
"checkpoint": "",
...
```

## Evaluation

`tools/test.py` script is designed for quality evaluation of instance
Text spotting models.

### PyTorch

For example, to evaluate text-spotting-0001 model on ICDAR2015 test dataset
using PyTorch backend run:

```bash
python tools/test.py \
    --prob_threshold 0.8 \
    --dataset IC15TEST_FIXED.json \
    --mean_pixel 123.675 116.28 103.53 \
    --std_pixel 58.395 57.12 57.375 \
    --rgb \
    --size 768 1280 \
  pytorch \
    --model configs/text-spotting-0001.json \
    --weights <path_to_checkpoint>.pth
```

> **Note:** Use `CUDA_VISIBLE_DEVICES` environment variable to configure which
(if any) GPUs to use during evaluation. If empty value is assigned, PyTorch uses
CPU backend.

## Demo

In order to see how trained model works using OpenVINO please refer to [Text Spotting Python* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/text_spotting_demo). Before running the demo you have to export trained model to IR. Please see below how to do that.

## Export PyTorch models to OpenVINO

To run the model via OpenVINO one has to export PyTorch model to ONNX first and
then convert it to OpenVINO Internal Representation (IR) using Model Optimizer.

Model will be split into three parts:
- Text detector (Mask-RCNN like)
- Additional text recognition head
  - Text recognition encoder
  - Text recognition decoder

### Export to ONNX

The `tools/convert_to_onnx.py` script exports a given model to ONNX representation.

```bash
python tools/convert_to_onnx.py \
    --model configs/text-spotting-0001.json \
    --ckpt <path_to_checkpoint>.pth \
    --input_size 768 1280 \
    --show_flops \
    --output_folder /tmp/output_folder
```


### Convert to IR


Conversion from ONNX model representation to OpenVINO IR is straightforward and
handled by OpenVINO Model Optimizer. Please refer to Model Optimizer
documentation for details on how it works.

1. text-spotting-0001-detector
```bash
mo.py \
    --model_name text-spotting-0001-detector \
    --input_shape="[1,3,768,1280],[1,3]" \
    --input=im_data,im_info \
    --mean_values="im_data[123.675,116.28,103.53]" \
    --scale_values="im_data[58.395000005673076,57.120000003655676,57.37500003220172],im_info[1]" \
    --output=boxes,scores,classes,raw_masks,text_features \
    --reverse_input_channels \
    --input_model /tmp/output_folder/detector.onnx
```
3. text-spotting-0001-encoder
```bash
mo.py \
    --model_name text-spotting-0001-encoder \
    --input_model /tmp/output_folder/encoder.onnx
```
3. text-spotting-0001-decoder
```bash
mo.py \
    --model_name text-spotting-0001-decoder \
    --input_model /tmp/output_folder/decoder.onnx
```
