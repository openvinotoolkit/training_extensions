# OpenVINO Training Extensions

OpenVINO Training Extensions provide a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.


# Quick Start Guide

## Setup OpenVINO Training Extensions

1. Clone repository in the working directory

  ```
  cd /<path_to_working_dir>
  git clone https://github.com/opencv/openvino_training_extensions.git
  ```

2. Install prerequisites

  ```
  sudo apt-get install libturbojpeg python3-tk python3-pip virtualenv
  ```


# Models

* [PyTorch](pytorch_toolkit)

  * [Action recognition](pytorch_toolkit/action_recognition)
  * [Face recognition](pytorch_toolkit/face_recognition)
  * [Person reidentification](pytorch_toolkit/person_reidentification)
  * [Human pose estimation](pytorch_toolkit/human_pose_estimation)
  * [Instance segmentation](pytorch_toolkit/instance_segmentation)
  * [Object Detection](pytorch_toolkit/object_detection)
    - [Face Detection](pytorch_toolkit/object_detection/face_detection.md)
    - [Person Vehicle Bike Detector](pytorch_toolkit/object_detection/person_vehicle_bike_detection.md)
  * [Segmentation of thoracic organs](pytorch_toolkit/segthor)
  * [Super resolution](pytorch_toolkit/super_resolution)

* [TensorFlow](tensorflow_toolkit)

  * [Action Detection](tensorflow_toolkit/action_detection)
  * [Image retrieval](tensorflow_toolkit/image_retrieval)
  * [License Plate Recognition](tensorflow_toolkit/lpr)
  * [Person Vehicle Bike Detector](tensorflow_toolkit/person_vehicle_bike_detector)
  * [SSD MobileNet FPN 602](tensorflow_toolkit/ssd_mobilenet_fpn_602)
  * [SSD Object Detection](tensorflow_toolkit/ssd_detector)
  * [Text detection](tensorflow_toolkit/text_detection)
  * [Text recognition](tensorflow_toolkit/text_recognition)
  * [Vehicle Attributes](tensorflow_toolkit/vehicle_attributes)

# Tools

* [PyTorch](pytorch_toolkit)

  * [Neural Networks Compression Framework](pytorch_toolkit/nncf)
