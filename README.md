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
  * [Action Recognition 2](pytorch_toolkit/action_recognition_2)
    - [Gesture Recognition](pytorch_toolkit/action_recognition_2/model_templates/gesture-recognition)
  * [Object Re-Identification](pytorch_toolkit/object_reidentification)
    - [Face Recognition](pytorch_toolkit/object_reidentification/face_recognition)
    - [Person Re-Identification](pytorch_toolkit/object_reidentification/person_reidentification)
    - [Vehicle Re-Identification](pytorch_toolkit/object_reidentification/vehicle_reidentification)
  * [Human Pose Estimation](pytorch_toolkit/human_pose_estimation)
  * [Instance Segmentation](pytorch_toolkit/instance_segmentation)
  * [Machine Translation](pytorch_toolkit/machine_translation)
  * [Object Detection](pytorch_toolkit/object_detection)
    - [Face Detection](pytorch_toolkit/object_detection/model_templates/face-detection)
    - [Horizontal Text Detection](pytorch_toolkit/object_detection/model_templates/horizontal-text-detection/)
    - [Person Detection](pytorch_toolkit/object_detection/model_templates/person-detection/)
    - [Person Vehicle Bike Detection](pytorch_toolkit/object_detection/model_templates/person-vehicle-bike-detection)
    - [Vehicle Detection](pytorch_toolkit/object_detection/model_templates/vehicle-detection)
  * [Eye State Classification](pytorch_toolkit/open_closed_eye)
  * [Segmentation of Thoracic Organs](pytorch_toolkit/segthor)
  * [Super Resolution](pytorch_toolkit/super_resolution)
  * [Text Spotting](pytorch_toolkit/text_spotting)
  * [Question Answering](pytorch_toolkit/question_answering)

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
  * [Bert](tensorflow_toolkit/bert)

---
\* Other names and brands may be claimed as the property of others.

# Web UI

OpenVINO™ Training Extensions provide [Web UI](web) for training models and annotating data in a convenient way using a graphical interface.
