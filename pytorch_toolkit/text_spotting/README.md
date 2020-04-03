# Text Spotting

This repository contains inference and training code for Text Spotting models based on Mask R-CNN like networks.
Models code is designed to enable ONNX\* export (with custom operations) and inference on CPU via OpenVINO™.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* GCC\* 7.4.0
* Python\* 3.6 or newer
* PyTorch\* (custom, see Installation section)
* CUDA\* 10.1
* OpenVINO™ 2020.1 with Python API

### Installation

Create and activate virtual environment:

```bash
bash init_venv.sh
```

> **NOTE** on this step will be install custom version of
> [Pytorch](https://github.com/Ilya-Krylov/pytorch/tree/enable_export_of_custom_onnx_operations_with_tuples_as_output)
> and [torchvison](https://github.com/pytorch/vision/tree/be6dd4720652d630e95d968be2a4e1ae62f8807e) from specific
> commit. For more information, see the [init_venv.sh](init_venv.sh)

### Download Datasets

To be able to train networks and/or get quality metrics for pre-trained ones,
one have to download one dataset at least.
* https://rrc.cvc.uab.es/ - ICDAR2013 (Focused Scene Text), ICDAR2013 (Incidental Scene Text), ICDAR2017 (MLT), ... .
* http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500) MSRA-TD500.
* https://bgshih.github.io/cocotext/ COCO-Text.

### Convert Datasets

Extract downloaded datasets in following images_folder:  `$WORK_DIR/openvino_training_extensions/pytorch_toolkit/text_spotting/data/coco`

Convert extracted datasets to format that is used internally.

```bash
python3 tools/create_dataset.py --config datasets/dataset_train.json --output data/coco/IC13TRAINTEST_IC15TRAIN_MSRATD500TRAINTEST_COCOTEXTTRAINVAL.json
python3 tools/create_dataset.py --config datasets/dataset_test.json --output data/coco/IC15TEST.json
```

Examples of dataset_configuration.json can be found in `openvino_training_extensions/pytorch_toolkit/text_spotting/datasets`.

The structure of the folder with datasets:
```
texxt_spotting/data/coco/
    ├── coco-text
    ├── icdar2013
    ├── icdar2015
    ├── MSRA-TD500
    ├── IC13TRAINTEST_IC15TRAIN_MSRATD500TRAINTEST_COCOTEXTTRAINVAL.json
    └── IC15TEST.json
```


## Training

To train Text Spotter model run:

```bash
python3 tools/train.py configs/text-spotting-0001.json
```

One can point to pre-trained model [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/text_spotter/model_step_200000.pth) inside configuration file to start training from pre-trained weights. Change `configs/text-spotting-0001.json`:
```
...
"checkpoint": "<path_to_weights>",
...
```

> **Known issue:** 'Nan' in log output.
> ```
> metrics/detection/cls_accuracy: 0.95204, metrics/rpn/cls_accuracy/0: 0.969265, metrics/rpn/cls_accuracy/1: 1.0,
> metrics/rpn/cls_accuracy/2: 1.0, metrics/rpn/cls_accuracy/3: nan, metrics/rpn/cls_accuracy/4: nan, metrics/rpn/cls_precision/0: nan,
> metrics/rpn/cls_precision/1: nan, metrics/rpn/cls_precision/2: nan, metrics/rpn/cls_precision/3: nan, metrics/rpn/cls_precision/4: nan,
> metrics/rpn/cls_recall/0: nan, metrics/rpn/cls_recall/1: nan, metrics/rpn/cls_recall/2: nan, metrics/rpn/cls_recall/3: nan,
> time elapsed/~left: 0:34:33 / 2 days, 7:37:14 (1.01 sec/it)
> WARNING 17-01-20 13:42:25 x2num.py:  14] NaN or Inf found in input tensor.
> ```

## Evaluation

`tools/test.py` script is designed for quality evaluation of instance
Text spotting models.

### PyTorch

For example, to evaluate text-spotting-0001 model on ICDAR2015 test dataset
using PyTorch backend run:

```bash
python tools/test.py \
    --prob_threshold 0.8 \
    --dataset IC15TEST.json \
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

In order to see how trained model works using OpenVINO™ please refer to [Text Spotting Python* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/text_spotting_demo). Before running the demo you have to export trained model to IR. Please see below how to do that.

## Export PyTorch Models to OpenVINO™

To run the model via OpenVINO™ one has to export PyTorch model to ONNX first and
then convert to OpenVINO™ Internal Representation (IR) using Model Optimizer.

Model will be split into three parts:
- Text detector (Mask-RCNN like)
- Additional text recognition head
  - Text recognition encoder
  - Text recognition decoder

### Export to ONNX*

The `tools/convert_to_onnx.py` script exports a given model to ONNX representation.

```bash
python tools/convert_to_onnx.py \
    --model configs/text-spotting-0001.json \
    --ckpt <path_to_checkpoint>.pth \
    --input_size 768 1280 \
    --show_flops \
    --output_folder export
```


### Convert to IR

Conversion from ONNX model representation to OpenVINO™ IR is straightforward and
handled by OpenVINO™ Model Optimizer. Please refer to [Model Optimizer
documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for details on how it works.

1. text-spotting-0001-detector:
    ```bash
    mo.py \
        --model_name text-spotting-0001-detector \
        --input_shape="[1,3,768,1280],[1,3]" \
        --input=im_data,im_info \
        --mean_values="im_data[123.675,116.28,103.53]" \
        --scale_values="im_data[58.395000005673076,57.120000003655676,57.37500003220172],im_info[1]" \
        --output=boxes,scores,classes,raw_masks,text_features \
        --reverse_input_channels \
        --input_model export/detector.onnx
    ```

3. text-spotting-0001-encoder:
    ```bash
    mo.py \
        --model_name text-spotting-0001-encoder \
        --input_model export/encoder.onnx
    ```

3. text-spotting-0001-decoder:
    ```bash
    mo.py \
        --model_name text-spotting-0001-decoder \
        --input_model export/decoder.onnx
    ```
