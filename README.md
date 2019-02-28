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
  cd openvino_training_extensions
  git submodule update --init --recursive
  ```

2. Install prerequisites

  ```
  sudo apt-get install libturbojpeg python3-tk python3-pip virtualenv 2to3
  ```


## Frameworks
* [Tensorflow](tensorflow_toolkit)
* [PyTorch](pytorch_toolkit)
