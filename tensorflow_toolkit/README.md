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

## Models
After installation, you are ready to train your own models, evaluate them, use
them for predictions.

* [Action Detection](action_detection)
* [Image retrieval](image_retrieval)
* [License Plate Recognition](lpr)
* [Person Vehicle Bike Detector](person_vehicle_bike_detector)
* [SSD MobileNet FPN 602](ssd_mobilenet_fpn_602)
* [SSD Object Detection](ssd_detector)
* [Text detection](text_detection)
* [Text recognition](text_recognition)
* [Vehicle Attributes](vehicle_attributes)
* [Image Retrieval](image_retrieval)
