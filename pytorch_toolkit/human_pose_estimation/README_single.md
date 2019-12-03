# Global Context for Convolutional Pose Machines

This repository contains training code for the paper [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf). The work improves original [convolutional pose machine](https://arxiv.org/pdf/1602.00134.pdf) architecture for artculated human pose estimation in both accuracy and inference speed. On the Look Into Person (LIP) test set this code achives 87.9% PCKh for a single model, the 2-stage version of this network runs with more than 160 frames per second on a GPU and ~20 frames per second on a CPU. The result can be reproduced using this repository.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Training on LIP dataset](#training-lip)
* [Training on COCO dataset](#training-coco)
* [Pre-trained model](#pre-trained-model)
* [OpenVINO demo](#openvino-demo)
* [Citation](#citation)

## Prerequisites

1. Create virtual environment `bash init_venv.sh`.
2. Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet) (sgd option) for training.

## Training on LIP dataset <a name="training-lip"/>

1. Download the [Look Into Person dataset](http://47.100.21.47:9999/overview.php) and unpack it to `<LIP_HOME>` folder.
2. To start training run in terminal:
```
python train_single.py --dataset-folder <LIP_HOME> --checkpoint-path mobilenet_sgd_68.848.pth.tar --from-mobilenet
```

#### Validation

* Run in terminal:
```
python val_single.py --dataset-folder <LIP_HOME> --checkpoint-path <CHECKPOINT> --name-dataset Lip
```
One should observe ~84% PCKh on validation set (use `--multiscale` and set `flip` to `True` for better results).

[OPTIONAL] Pass `--visualize` key to see predicted keypoints results.

The final number on the test set was obtained with addition of validation data into training.

## Training on COCO dataset <a name="training-coco"/>

1. Download COCO 2017 dataset: [http://cocodataset.org/#download](http://cocodataset.org/#download) (train, val, annotations) and unpack it to `<COCO_HOME>` folder.
2. Convert train annotations in internal format. Run in terminal:
```
python scripts/convert_coco_labels.py --labels-path <COCO_HOME>/annotations/person_keypoints_train2017.json --output-name <COCO_HOME>/annotations/person_keypoints_converted_all.json
```
3. To start training run in terminal:
```
python train_single_coco.py --dataset-folder <COCO_HOME> --checkpoint-path mobilenet_sgd_68.848.pth.tar --from-mobilenet
```

#### Validation

* Run in terminal:
```
python val_single.py --dataset-folder <COCO_HOME> --checkpoint-path <CHECKPOINT>` --name-dataset CocoSingle
```
One should observe ~68% mAP on validation set.

[OPTIONAL] Pass `--visualize` key to see predicted keypoints results.

## Pre-trained model <a name="pre-trained-model"/>

Pre-trained on COCO dataset model is available at: https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/single-human-pose-estimation-0001.pth.

#### Conversion to OpenVINO format:

1. Convert PyTorch model to ONNX format: run script in terminal

```
python scripts/convert_to_onnx.py --checkpoint-path <CHECKPOINT> \
    --single-person \
    --input-size 256 256
```

(for model trained on COCO dataset use`--input-size 384 288`). It produces `human-pose-estimation.onnx`.

2. Convert ONNX model to OpenVINO format with Model Optimizer. For model trained on LIP dataset run in terminal:

```
python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation.onnx \
    --input data \
    --mean_values data[128.0,128.0,128.0] \
    --scale_values data[256] \
    --output stage_1_output_1_heatmaps
```

This produces model `human-pose-estimation.xml` and weights `human-pose-estimation.bin` in single-precision floating-point format (FP32). For model trained on COCO dataset run in terminal:

```
python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation.onnx  \
    --input_shape [1,3,384,288]  \
    --input data \
    --mean_values data[123.675,116.28,103.53] \
    --scale_values data[58.395,57.12,57.375] \
    --output stage_4_output_1_heatmaps \
    --reverse_input_channels
```

## OpenVINO demo <a name="openvino-demo"/>

OpenVINO provides multi-person pose estimation demo, which is able to use these models as pose estimation networks. Please, see details in the demo [readme](https://github.com/opencv/open_model_zoo/tree/demos/python_demos/single_human_pose_estimation_demo/README.md).

## Citation:

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
