<div align="center">

# OpenVINO™ Training Extensions

---

[Key Features](#key-features) •
[Installation](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/installation.html) •
[Documentation](https://openvinotoolkit.github.io/training_extensions/latest/index.html) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/otx)](https://pypi.org/project/otx)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.8%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-1.13.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2023.3-purple)]()

<!-- markdownlint-enable  MD042 -->

[![Codecov](https://codecov.io/gh/openvinotoolkit/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/openvinotoolkit/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/openvinotoolkit/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/openvinotoolkit/training_extensions)
[![Pre-Merge Test](https://github.com/openvinotoolkit/training_extensions/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/pre_merge.yml)
[![Nightly Test](https://github.com/openvinotoolkit/training_extensions/actions/workflows/daily.yml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/daily.yml)
[![Build Docs](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/otx?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/otx)

---

</div>

## Introduction

OpenVINO™ Training Extensions is a low-code transfer learning framework for Computer Vision.
The CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field.
OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on [PyTorch](https://pytorch.org) and [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

OpenVINO™ Training Extensions provides a "model template" for every supported task type, which consolidates necessary information to build a model.
Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general.
If you are an experienced user, you can configure your own model based on [torchvision](https://pytorch.org/vision/stable/index.html), [mmcv](https://github.com/open-mmlab/mmcv), [timm](https://github.com/huggingface/pytorch-image-models) and [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo).

Furthermore, OpenVINO™ Training Extensions provides automatic configuration for ease of use.
The framework will analyze your dataset and identify the most suitable model and figure out the best input size setting and other hyper-parameters.
The development team is continuously extending this [Auto-configuration](https://openvinotoolkit.github.io/training_extensions/stable/guide/explanation/additional_features/auto_configuration.html) functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

### Key Features

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box support
- **Semantic segmentation**
- **Instance segmentation** including tiling algorithm support
- **Action recognition** including action classification and detection
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

OpenVINO™ Training Extensions supports the [following learning methods](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/algorithms/index.html):

- **Supervised**, incremental training, which includes class incremental scenario and contrastive learning for classification and semantic segmentation tasks
- **Semi-supervised learning**
- **Self-supervised learning**

OpenVINO™ Training Extensions provides the following usability features:

- [Auto-configuration](https://openvinotoolkit.github.io/training_extensions/stable/guide/explanation/additional_features/auto_configuration.html). OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model with appropriate input size to provide the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.
- [Datumaro](https://openvinotoolkit.github.io/datumaro/stable/index.html) data frontend: OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. We are constantly working to extend supported formats to give more freedom of datasets format choice.
- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Mixed-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient [hyper-parameter optimization module (HPO)](https://openvinotoolkit.github.io/training_extensions/stable/guide/explanation/additional_features/hpo.html). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.

---

## Getting Started

### Installation

Please refer to the [installation guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/installation.html).

Note: Python 3.8, 3.9 and 3.10 were tested, along with Ubuntu 18.04, 20.04 and 22.04.

### OpenVINO™ Training Extensions CLI Commands

- `otx find` helps you quickly find the best pre-configured models templates as well as a list of supported backbones
- `otx build` creates the workspace folder with all necessary components to start training. It can help you configure your own model with any supported backbone and even prepare a custom split for your dataset
- `otx train` actually starts training on your dataset
- `otx eval` runs evaluation of your trained model in PyTorch or OpenVINO™ IR format
- `otx optimize` runs an optimization algorithm to quantize and prune your deep learning model with help of [NNCF](https://github.com/openvinotoolkit/nncf) and [POT](https://docs.openvino.ai/latest/pot_introduction.html) tools.
- `otx export` starts exporting your model to the OpenVINO™ IR format
- `otx deploy` outputs the exported model together with the self-contained python package, a demo application to port and infer it outside of this repository.
- `otx demo` allows one to apply a trained model on the custom data or the online footage from a web camera and see how it will work in a real-life scenario.
- `otx explain` runs explain algorithm on the provided data and outputs images with the saliency maps to show how your model makes predictions.

You can find more details with examples in the [CLI command intro](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/cli_commands.html).

---

## Updates

### v1.5.0 (4Q23)

- Enable configurable confidence threshold for otx eval and export (<https://github.com/openvinotoolkit/training_extensions/pull/2388>)
- Add YOLOX variants as new object detector models (<https://github.com/openvinotoolkit/training_extensions/pull/2402>)
- Enable FeatureVectorHook to support action tasks (<https://github.com/openvinotoolkit/training_extensions/pull/2408>)
- Add ONNX metadata to detection, instance segmantation, and segmentation models (<https://github.com/openvinotoolkit/training_extensions/pull/2418>)
- Add a new feature to configure input size (<https://github.com/openvinotoolkit/training_extensions/pull/2420>)
- Introduce the OTXSampler and AdaptiveRepeatDataHook to achieve faster training at the small data regime (<https://github.com/openvinotoolkit/training_extensions/pull/2428>)
- Add a new object detector Lite-DINO (<https://github.com/openvinotoolkit/training_extensions/pull/2457>)
- Add Semi-SL Mean Teacher algorithm for Instance Segmentation task (<https://github.com/openvinotoolkit/training_extensions/pull/2444>)
- Official supports for YOLOX-X, YOLOX-L, YOLOX-S, ResNeXt101-ATSS (<https://github.com/openvinotoolkit/training_extensions/pull/2485>)
- Add new argument to track resource usage in train command (<https://github.com/openvinotoolkit/training_extensions/pull/2500>)
- Add Self-SL for semantic segmentation of SegNext families (<https://github.com/openvinotoolkit/training_extensions/pull/2215>)
- Adapt input size automatically based on dataset statistics (<https://github.com/openvinotoolkit/training_extensions/pull/2499>)

### Release History

Please refer to the [CHANGELOG.md](CHANGELOG.md)

---

## Branches

- [develop](https://github.com/openvinotoolkit/training_extensions/tree/develop)
  - Mainly maintained branch for developing new features for the future release
- [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc)
  - Previously developed models can be found on this branch

---

## License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---

## Issues / Discussions

Please use [Issues](https://github.com/openvinotoolkit/training_extensions/issues/new/choose) tab for your bug reporting, feature requesting, or any questions.

---

## Known limitations

[misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch contains training, evaluation, and export scripts for models based on TensorFlow and PyTorch.
These scripts are not ready for production. They are exploratory and have not been validated.

---

## Disclaimer

Intel is committed to respecting human rights and avoiding complicity in human rights abuses.
See Intel's [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html).
Intel's products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

---
