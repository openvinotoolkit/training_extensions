# OpenVINO™ Training Extensions

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Quick Start Guide

### Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.6+
* [OpenVINO™](https://software.intel.com/en-us/openvino-toolkit) - for exporting and running models
* [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) - for training on GPU

### Setup OpenVINO™ Training Extensions

1. Clone repository in the working directory by running the following:
    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    export OTE_DIR=`pwd`/training_extensions
    ```

2. Clone Open Model Zoo repository to run demos:
    ```
    git clone https://github.com/openvinotoolkit/open_model_zoo --branch develop
    export OMZ_DIR=`pwd`/open_model_zoo
    ```

3. Install prerequisites by running the following:
    ```
    sudo apt-get install python3-pip virtualenv
    ```

4. Create and activate virtual environment:
    ```
    cd training_extensions
    virtualenv venv
    source venv/bin/activate
    ```

6. Install `ote` package:
    ```
    pip3 install -e ote/
    ```

## Models

After installation, you are ready to train your own models, evaluate and use
them for prediction.

* [Action Recognition](models/action_recognition)
  - [Custom Action Recognition](models/action_recognition/model_templates/custom-action-recognition)
  - [Gesture Recognition](models/action_recognition/model_templates/gesture-recognition)
* [Instance Segmentation](models/instance_segmentation)
  - [COCO instance segmentation](models/instance_segmentation/model_templates/coco-instance-segmentation)
  - [Custom instance segmentation](models/instance_segmentation/model_templates/custom-instance-segmentation)
* [Image classification](models/image_classification)
  - [Custom image classification](models/image_classification/model_templates/custom-classification)
* [Object Detection](models/object_detection)
  - [Custom Object Detection](models/object_detection/model_templates/custom-object-detection)
  - [Face Detection](models/object_detection/model_templates/face-detection)
  - [Horizontal Text Detection](models/object_detection/model_templates/horizontal-text-detection)
  - [Person Detection](models/object_detection/model_templates/person-detection)
  - [Person Vehicle Bike Detection](models/object_detection/model_templates/person-vehicle-bike-detection)
  - [Vehicle Detection](models/object_detection/model_templates/vehicle-detection)
* [Text Spotting](models/text_spotting)
  - [Alphanumeric Text Spotting](models/text_spotting/model_templates/alphanumeric-text-spotting)

## Optimization

The image classification and object detection models can be compressed
by [NNCF](https://github.com/openvinotoolkit/nncf) framework.

See details in the corresponding readme files of the models.

## Misc

Models that were previously developed can be found [here](misc).

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) before starting work on a pull request.

## Known Limitations

Currently, training, exporting, evaluation scripts for TensorFlow\*-based models and the most of PyTorch\*-based models from [Misc](#misc) section are exploratory and are not validated.

---
\* Other names and brands may be claimed as the property of others.
