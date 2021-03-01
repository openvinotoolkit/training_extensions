# OpenVINO™ Training Extensions

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

# Quick Start Guide

## Setup OpenVINO™ Training Extensions

0. Download and install [OpenVINO™](https://software.intel.com/en-us/openvino-toolkit).

1. Clone repository in the working directory by running the following:

    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    ```

2. Install prerequisites by running the following:

    ```
    sudo apt-get install python3-pip virtualenv
    ```

3. Create and activate virtual environment:
    ```
    virtualenv venv
    . venv/bin/activate
    ```

4. Install `ote` package:
    ```
    pip3 install -e ote/
    ```

# Models

After installation, you are ready to train your own models, evaluate and use
them for prediction.

* [Action Recognition](models/action_recognition_2)
  - [Gesture Recognition](models/action_recognition_2/model_templates/gesture-recognition)
* [Instance Segmentation](models/instance_segmentation)
  - [COCO instance segmentation](models/instance_segmentation/model_templates/coco-instance-segmentation/readme.md)
* [Image classification](models/image_classification)
  - [Custom image classification](models/image_classification/model_templates/custom-classification/README.md)
* [Object Detection](models/object_detection)
  - [Face Detection](models/object_detection/model_templates/face-detection)
  - [Horizontal Text Detection](models/object_detection/model_templates/horizontal-text-detection/)
  - [Person Detection](models/object_detection/model_templates/person-detection/)
  - [Person Vehicle Bike Detection](models/object_detection/model_templates/person-vehicle-bike-detection)
  - [Vehicle Detection](models/object_detection/model_templates/vehicle-detection)
* [Text Spotting](models/text_spotting)
  - [Alphanumeric Text Spotting](models/text_spotting/model_templates/alphanumeric-text-spotting/readme.md)

# Misc

Models that were previously developed can be found [here](misc/README.md).

# Web UI

OpenVINO™ Training Extensions provide [Web UI](web) for training models and annotating data in a convenient way using a graphical interface.
