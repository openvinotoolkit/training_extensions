# Training Toolbox for PyTorch*

Training Toolbox for PyTorch\* provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Prerequisites

- Ubuntu\* 16.04 / 18.04
- Python\* 3.6+
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python prerequisites, refer to `requirements.txt`
- *(Optional)* [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  to export trained models


### Models

After installation, you are ready to train your own models, evaluate and use
them for prediction.

* [Action Recognition](action_recognition)
* [ASL Recognition](asl_recognition)
* [Object Re-Identification](object_reidentification)
  - [Face Recognition](object_reidentification/face_recognition)
  - [Person Re-Identification](object_reidentification/person_reidentification)
  - [Vehicle Re-Identification](object_reidentification/vehicle_reidentification)
* [Human Pose Estimation](human_pose_estimation)
* [Instance Segmentation](instance_segmentation)
* [Object Detection](object_detection)
  - [Face Detection](object_detection/model_templates/face-detection)
  - [Horizontal Text Detection](object_detection/model_templates/horizontal-text-detection)
  - [Person Detection](object_detection/model_templates/person-detection)
  - [Person Vehicle Bike Detection](object_detection/model_templates/person-vehicle-bike-detection)
  - [Vehicle Detection](object_detection/model_templates/vehicle-detection)
* [Eye State Classification](open_closed_eye)
* [Segmentation of Thoracic Organs](segthor)
* [Super Resolution](super_resolution)
* [Formula recognition](formula_recognition)
* [Text Spotting](text_spotting)


### Tools

Tools are intended to perform manipulations with trained models, like compressing models using Quantization-aware training or sparsity.

* [Neural Networks Compression Framework](nncf)

### Tests

In order to run tests please execute following commands:

```bash
pip3 install -e ote
python3 tests/run_model_templates_tests.py
```
