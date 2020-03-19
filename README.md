# OpenVINO™ Training Extensions

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

# Quick Start Guide

## Setup OpenVINO™ Training Extensions

1. Clone repository in the working directory by running the following:

    ```
    git clone https://github.com/opencv/openvino_training_extensions.git
    cd openvino_training_extensions
    ```

2. Install prerequisites by running the following:

    ```
    sudo apt-get install libturbojpeg python3-tk python3-pip virtualenv
    ```


# Models

* [PyTorch\*](pytorch_toolkit)

  * [Action Recognition](pytorch_toolkit/action_recognition)
  * [ASL Recognition](pytorch_toolkit/asl_recognition)
  * [Face Recognition](pytorch_toolkit/face_recognition)
  * [Person Reidentification](pytorch_toolkit/person_reidentification)
  * [Human Pose Estimation](pytorch_toolkit/human_pose_estimation)
  * [Instance Segmentation](pytorch_toolkit/instance_segmentation)
  * [Object Detection](pytorch_toolkit/object_detection)
    - [Face Detection](pytorch_toolkit/object_detection/face_detection.md)
    - [Person Vehicle Bike Detection](pytorch_toolkit/object_detection/person_vehicle_bike_detection.md)
  * [Eye State Classification](pytorch_toolkit/open_closed_eye)
  * [Segmentation of Thoracic Organs](pytorch_toolkit/segthor)
  * [Super Resolution](pytorch_toolkit/super_resolution)
  * [Text Spotting](pytorch_toolkit/text_spotting)

* [TensorFlow\*](tensorflow_toolkit)

  * [Action Detection](tensorflow_toolkit/action_detection)
  * [Image Retrieval](tensorflow_toolkit/image_retrieval)
  * [License Plate Recognition](tensorflow_toolkit/lpr)
  * [Person Vehicle Bike Detector](tensorflow_toolkit/person_vehicle_bike_detector)
  * [SSD MobileNet FPN 602](tensorflow_toolkit/ssd_mobilenet_fpn_602)
  * [SSD Object Detection](tensorflow_toolkit/ssd_detector)
  * [Text Detection](tensorflow_toolkit/text_detection)
  * [Text Recognition](tensorflow_toolkit/text_recognition)
  * [Vehicle Attributes](tensorflow_toolkit/vehicle_attributes)

# Tools

* [PyTorch\*](pytorch_toolkit)

  * [Neural Networks Compression Framework](pytorch_toolkit/nncf)

---
\* Other names and brands may be claimed as the property of others.
