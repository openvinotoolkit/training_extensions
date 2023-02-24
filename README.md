<div align="center">

# OpenVINO™ Training Extensions

---

[![python](https://img.shields.io/badge/python-3.8%2B-green)]()
[![openvino](https://img.shields.io/badge/openvino-2022.3.0-purple)]()
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f9ba89f9ea2a47eeb9d52c2acc311e6c)](https://www.codacy.com/gh/openvinotoolkit/training_extensions/dashboard?utm_source=github.com&utm_medium=referral&utm_content=openvinotoolkit/training_extensions&utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/openvinotoolkit/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/openvinotoolkit/training_extensions)
[![Build Docs](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

</div>

## Overview

OpenVINO™ Training Extensions is a low-code transfer learning framework for Computer Vision. OpenVINO™ Training Extensions lets users train, infer, optimize and deploy models simply and fast even with low expertise in the deep learning field. OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on [PyTorch](https://pytorch.org) and [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit). OpenVINO™ Training Extensions provides "model template" for every supported task type, which consolidates neccesary information to build a model. Model templates are validated on various datasets and serve one-stop shop for obtaining best models in general. If you are an experienced user, you can configure your own model based on [torchvision](https://pytorch.org/vision/stable/index.html), [pytorchcv](https://github.com/osmr/imgclsmob), [mmcv](https://github.com/open-mmlab/mmcv) and [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo). Moreover, OpenVINO™ Training Extensions provides automatic configuration of task types and hyperparameters. The framework will identify the most suitable model template based on your dataset, and choose the best hyperparameter configuration. The development team is continuously extending functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box support
- **Semantic segmentation**
- **Instance segmentation** including tiling algorithm support
- **Action recognition** including action classification and detection
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

OpenVINO™ Training Extensions supports the [following learning methods](https://openvinotoolkit.github.io/training_extensions/guide/explanation/algorithms/index.html):

- **Supervised**, incremental training including class incremental scenario and contrastive learning for classification and semantic segmentation tasks
- **Semi-supervised learning**
- **Self-supervised learning**

OpenVINO™ Training Extensions will provide the following features in coming releases:

- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Half-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient [hyper-parameter optimization module (HPO)](https://openvinotoolkit.github.io/training_extensions/guide/explanation/additional_features/hpo.html). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- OpenVINO™ Training Extensions uses [Datumaro](https://openvinotoolkit.github.io/datumaro/docs/) as the backend to hadle datasets. Thanks to that, OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. We constantly working to extend supported formats to give more freedom of datasets format choice.
- [Auto-configuration functionality](https://openvinotoolkit.github.io/training_extensions/guide/explanation/additional_features/auto_configuration.html). OpenVINO™ Training Extensions analyzes provided dataset and chooses the proper task and model template to have the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.

---

## OpenVINO™ Training Extensions CLI Commands

- `otx find` helps you quickly find the best pre-configured models templates as well as a list of supported backbones
- `otx build` creates the workspace folder with all necessary components to start training. It can help you configure your own model with any supported backbone and even prepare a custom split for your dataset
- `otx train` actually starts training on your dataset
- `otx eval` runs evaluation of your trained model in PyTorch or OpenVINO™ IR format
- `otx optimize` runs an optimization algorithm to quantize and prune your deep learning model with help of [NNCF](https://github.com/openvinotoolkit/nncf) and [POT](https://docs.openvino.ai/latest/pot_introduction.html) tools.
- `otx export` starts exporting your model to the OpenVINO™ IR format
- `otx deploy` outputs the exported model together with the self-contained python package, a demo application to port and infer it outside of this repository.
- `otx demo` allows one to apply a trained model on the custom data or the online footage from a web camera and see how it will work in a real-life scenario.
- `otx explain` runs explain algorithm on the provided data and outputs images with the saliency maps to show how your model makes predictions.

---

## Roadmap

### v1.0.0 (1Q23)

- Package Installation via PyPI
  - OpenVINO™ Training Extensions installation will be supported via PyPI
- CLI update
  - Update `find` command to find configurations of tasks/algorithms
  - Introduce `build` command to customize task or model configurations
  - Automatic algorihm selection for the `train` command using the given input dataset
- Adaptation of [Datumaro](https://github.com/openvinotoolkit/datumaro) component as a dataset interface
- Integrate hyper-parameter optimizations
- Support action recognition task

### v1.1.0 (2Q23)

- SDK/API update

---

## Repository

- Components
  - [OpenVINO™ Training Extensions API](otx/api)
  - [OpenVINO™ Training Extensions CLI](otx/cli)
  - [OpenVINO™ Training Extensions Algorithms](otx/algorithms)
- Branches
  - [develop](https://github.com/openvinotoolkit/training_extensions/tree/develop)
    - Mainly maintained branch for releasing new features in the future
  - [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc)
    - Previously developed models can be found on this branch

---

# Quick start guide

In order to get started with OpenVINO™ Training Extensions, see [the quick-start guide](QUICK_START_GUIDE.md).

---

# Documentation

Refer to our [documentation](https://openvinotoolkit.github.io/training_extensions/index.html) to read about explanation of the algorithms, additional features and also look into our dedicated tutorials covering all the functionality

---

# License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---

## Issues / Discussions

Please use [Issues](https://github.com/openvinotoolkit/training_extensions/issues/new/choose) tab for your bug reporting, feature requesting, or any questions.

---

## Known limitations

[misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch contains training, evaluation, and export scripts for models based on TensorFlow and PyTorch. These scripts are not ready for production. They are exploratory and have not been validated.

---
