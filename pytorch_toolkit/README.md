# Training Toolbox for PyTorch*

Training Toolbox for PyTorch\* provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Prerequisites

- Ubuntu\* 16.04 / 18.04
- Python\* 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python prerequisites, refer to `requirements.txt`
- *(Optional)* [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  to export trained models


### Models

After installation, you are ready to train your own models, evaluate and use
them for prediction.

* [Action Recognition](action_recognition)
* [ASL Recognition](asl_recognition)
* [Face Recognition](face_recognition)
* [Human Pose Estimation](human_pose_estimation)
* [Instance Segmentation](instance_segmentation)
* [Object Detection](object_detection)
  - [Face Detection](object_detection/face_detection.md)
  - [Person Vehicle Bike Detection](object_detection/person_vehicle_bike_detection.md)
* [Eye State Classification](open_closed_eye)
* [Segmentation of Thoracic Organs](segthor)
* [Super Resolution](super_resolution)
* [Text Spotting](text_spotting)


### Tools

Tools are intended to perform manipulations with trained models, like compressing models using Quantization-aware training or sparsity.

* [Neural Networks Compression Framework](nncf)
