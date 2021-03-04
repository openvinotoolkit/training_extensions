# Global Context for Convolutional Pose Machines

This repository contains training code for the [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf) paper. The work improves original [convolutional pose machine](https://arxiv.org/pdf/1602.00134.pdf) architecture for articulated human pose estimation in both accuracy and inference speed. On the Look Into Person (LIP) test set this code achieves 87.9% PCKh for a single model, the 2-stage version of this network runs with more than 160 frames per second on a GPU and about 20 frames per second on a CPU. The result can be reproduced using this repository.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Pretrained model](#pretrained-model)
* [Training on LIP dataset](#training-lip)
* [Training on COCO dataset](#training-coco)
* [OpenVINO demo](#openvino-demo)
* [Citation](#citation)

## Prerequisites

1. Create virtual environment: `bash init_venv.sh`.

2. [Download pretrained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar`](https://github.com/marvis/pytorch-mobilenet) (sgd option) for training.

## Pretrained model <a name="pretrained-model"/>

[Pretrained on the COCO dataset model](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/single-human-pose-estimation-0001.pth)

## Training on the LIP Dataset <a name="training-lip"/>

### Training

1. Download the [Look Into Person dataset (2.1 Single Person)](http://sysu-hcp.net/lip/overview.php) and unpack it to the `<LIP_HOME>` folder.

2. To start training, run in the terminal:

    ```bash
    python train_single.py \
        --dataset-folder <LIP_HOME> \
        --checkpoint-path mobilenet_sgd_68.848.pth.tar \
        --from-mobilenet
    ```

### Validation

Run in the terminal:

```bash
python val_single.py \
    --dataset-folder <LIP_HOME> \
    --checkpoint-path <CHECKPOINT> \
    --name-dataset Lip
```

One should observe about 84% PCKh on validation set (use `--multiscale` and set `flip` to `True` for better results).

*Optional*: Pass the `--visualize` key to see predicted keypoints results.

The final number on the test set was obtained with addition of validation data into training.

### Conversion to OpenVINO™ format

1. Convert PyTorch\* model to ONNX\* format: run script in terminal:

    ```bash
    python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT> \
        --single-person \
        --output-name single-human-pose-estimation.onnx \
        --input-size 256 256 \
        --mode-interpolation nearest \
        --num-refinement-stages 1
    ```

2. Convert the ONNX\* model to OpenVINO™ format with the Model Optimizer:

    ```bash
    python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py \
        --input_model single-human-pose-estimation.onnx \
        --input data \
        --mean_values data[128.0,128.0,128.0] \
        --scale_values data[256] \
        --output stage_1_output_1_heatmaps
    ```

    This produces model `human-pose-estimation.xml` and weights `human-pose-estimation.bin` in the single-precision floating-point format (FP32).

## Training on the COCO Dataset <a name="training-coco"/>

### Training

1. Download the [COCO 2017 dataset](http://cocodataset.org/#download) (train, val, annotations) and unpack it to the`<COCO_HOME>` folder.

2. Convert train annotations into the internal format:

    ```bash
    python scripts/convert_coco_labels.py \
        --labels-path <COCO_HOME>/annotations/person_keypoints_train2017.json \
        --output-name <COCO_HOME>/annotations/person_keypoints_train2017_converted.json
    ```

3. To start training, run in the terminal:

    ```bash
    python train_single_coco.py \
        --dataset-folder <COCO_HOME> \
        --checkpoint-path mobilenet_sgd_68.848.pth.tar \
        --from-mobilenet
    ```

    To use pretrained model as init checkpoint use follow command:

    ```bash
    python train_single_coco.py \
        --dataset-folder <COCO_HOME> \
        --checkpoint-path single-human-pose-estimation-0001.pth \
        --weights-only
    ```

### Validation

* Run in the terminal:

    ```bash
    python val_single.py \
        --dataset-folder <COCO_HOME> \
        --checkpoint-path <CHECKPOINT> \
        --name-dataset CocoSingle
    ```

You should observe about 68% mAP on the validation set.

*Optional*: Pass the `--visualize` key to see predicted keypoints results.


### Conversion to OpenVINO™ format

1. Convert PyTorch model to ONNX format:

    ```bash
    python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT> \
        --single-person \
        --output-name single-human-pose-estimation.onnx \
        --input-size 384 288 \
        --mode-interpolation nearest \
        --num-refinement-stages 4
    ```

2. Convert the ONNX\* model to OpenVINO™ format with the Model Optimizer:

    ```bash
    python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py \
        --input_model single-human-pose-estimation.onnx \
        --input_shape [1,3,384,288] \
        --input data \
        --mean_values data[123.675,116.28,103.53] \
        --scale_values data[58.395,57.12,57.375] \
        --output stage_4_output_1_heatmaps \
        --reverse_input_channels
    ```

## OpenVINO™ demo <a name="openvino-demo"/>

OpenVINO™ provides multi-person pose estimation demo, which is able to use these models as pose estimation networks. See details in the [demo](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/demos/single_human_pose_estimation_demo/python).

## Citation

If this helps your research, please cite the papers:

```
@inproceedings{osokin2019global_context_cpm,
    author={Osokin, Daniil},
    title={Global Context for Convolutional Pose Machines},
    booktitle = {arXiv preprint arXiv:1906.04104},
    year = {2019}
}

@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```
