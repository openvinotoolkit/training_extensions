# Training Toolbox for TensorFlow

Training Toolbox for TensorFlow provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Pre-requisites

- Ubuntu 16.04 / 18.04
- Python 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python pre-requisites refer to `requirements.txt`
- (Optional) [TensorFlow GPU pre-requisites](https://www.tensorflow.org/install/gpu).
- (Optional) [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  for exporting of the trained models

## Setup

1. Install requirements
  ```Bash
  pip3 install cython
  sudo apt install 2to3 protobuf-compiler
  ```

1. Download submodules
  ```Bash
  cd openvino_training_extensions
  git submodule update --init --recommend-shallow external/cocoapi external/models
  ```

2. Compile Protobuf libraries
  ```Bash
  cd openvino_training_extensions/external/models/research/
  protoc object_detection/protos/*.proto --python_out=.
  ```

## Models
After installation, you are ready to train your own models, evaluate them, use
them for predictions.

* [License Plate Recognition](lpr)
* [Person Vehicle Bike Detector](person_vehicle_bike_detector)
* [SSD Object Detection](ssd_detector)
* [Vehicle Attributes](vehicle_attributes)
