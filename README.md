# Quick Start Guide

Training Toolbox for TensorFlow provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Pre-requisites

- Ubuntu 16.04
- Python 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python pre-requisites refer to `requirements.txt`
- (Optional) [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  for exporting of the trained models

## Setup Training Toolbox for TensorFlow

1. Clone repository in the working directory

    ```
    cd /<path_to_working_dir>
    git clone https://github.com/opencv/training_toolbox_tensorflow.git
    cd training_toolbox_tensorflow
    git submodule update --init --recursive
    ```

2. Install prerequisites

    ```
    sudo apt-get install libturbojpeg
    sudo apt-get install python3-tk
    ```

3. Create virtual environment

    ```
    sh tools/init_venv.sh
    ```

4. Start to work

    ```
    cd /<path_to_working_dir>/training_toolbox_tensorflow
    . venv/bin/activate
    ```

## Models
After installation, you are ready to train your own models, evaluate them, use
them for predictions.

[SSD Object Detection](models/ssd_detector/README.md)
