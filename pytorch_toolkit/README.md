# Training Toolbox for PyTorch*

Training Toolbox for PyTorch\* provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Prerequisites

- Ubuntu\* 16.04 / 18.04
- Python\* 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python prerequisites, refer to `requirements.txt`
- *(Optional)* [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  to export trained models

## Quick Start Guide

### Setup Training Toolbox for PyTorch

1. <a name="create-venv"/>Create virtual environment</a>:

    ```
    cd /<path_to_working_dir>/training_toolbox/pytorch_toolkit/<model>
    bash init_venv.sh
    ```

2. Start working:

    ```
    . venv/bin/activate
    ```

    >**NOTE**: If you have installed the OpenVINO&trade; toolkit after creating
    a virtual environment, [recreate one](#create_venv)
    to install required packages for the Model Optimizer into a single virtual environment.

> **NOTE**: Update several environment variables required to compile and run OpenVINO™ toolkit applications, for details see [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).


### Models

After installation, you are ready to train your own models, evaluate and use
them for prediction.

* [Action Recognition](action_recognition)
* [Face Recognition](face_recognition)
* [Human Pose Estimation](human_pose_estimation)
* [Instance Segmentation](instance_segmentation)
* [Object Detection](object_detection)
  - [Face Detection](object_detection/face_detection.md)
  - [Person Vehicle Bike Detector](object_detection/person_vehicle_bike_detection.md)
* [Segmentation of Thoracic Organs](segthor)
* [Super Resolution](super_resolution)

### Tools

Tools are intended to perform manipulations with trained models, like compressing models using Quantization-aware training or sparsity.

* [Neural Networks Compression Framework](nncf)

