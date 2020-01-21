# Training Toolbox for TensorFlow*

Training Toolbox for TensorFlow\* provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Prerequisites

- Ubuntu\* 16.04 / 18.04
- Python\* 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python prerequisites, refer to `requirements.txt`
- *(Optional)* [TensorFlow GPU prerequisites](https://www.tensorflow.org/install/gpu).
- *(Optional)* [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  to export trained models

## Setup

Install requirements:
  ```bash
  pip3 install cython
  sudo apt install 2to3 protobuf-compiler
  ```

## Models

After installation, you are ready to train your own models, evaluate them, and use
them for predictions.

* [Action Detection](action_detection)
* [Image Retrieval](image_retrieval)
* [License Plate Recognition](lpr)
* [Person Vehicle Bike Detector](person_vehicle_bike_detector)
* [SSD MobileNet FPN 602](ssd_mobilenet_fpn_602)
* [SSD Object Detection](ssd_detector)
* [Text Detection](text_detection)
* [Text Recognition](text_recognition)
* [Vehicle Attributes](vehicle_attributes)
* [Image Retrieval](image_retrieval)
