<div align="center">

<img src="https://raw.githubusercontent.com/openvinotoolkit/training_extensions/develop/docs/source/_static/logos/otx-logo-black.png" width="200px">

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

OpenVINO™ Training Extensions (OTX) is command-line interface (CLI) framework designed for low-code deep learning model training. OTX lets developers train/inference/optimize models with a diverse combination of model architectures and learning methods using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit). For example, users can train a ResNet18-based SSD ([Single Shot Detection](https://arxiv.org/abs/1512.02325)) model in a semi-supervised manner without worrying about setting a configuration manually. `otx build` and `otx train` commands will automatically analyze users' dataset and do necessary tasks for training the model with best configuration. OTX provides the following features:

- Provide a set of pre-configured models for quick start
  - `otx find` helps you quickly finds the best pre-configured models for common task types like classification, detection, segmentation, and anomaly analysis.
- Configure and train a model from torchvision, [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo)
  - `otx build` can help you configure your own model based on torchvision and OpenVINO Model Zoo models. You can replace backbones, necks and heads for your own preference (Currently only backbones are supported).
- Provide several learning methods including supervised, semi-supervised, imbalanced-learn, class-incremental, self-supervised representation learning
  - `otx build` helps you automatically identify the best learning methods for your data and model. All you need to do is to set your data in the supported format. If you don't specify a model, the framework will automatically sets the best model for you. For example, if your dataset has long-tailed and partially-annotated bounding box annotations, OTX auto-configurator will choose a semi-supervised imbalanced-learning method and an appropriate model with the best parameters.
- Integrated efficient hyper-parameter optimization
  - OTX has an integrated, efficient hyper-parameter optimization module. So, you don't need to worry about searching right hyper-parameters. Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- Support widely-used annotation formats
  - OTX uses [Datumaro](https://github.com/openvinotoolkit/datumaro), which is designed for dataset building and transformation, as a default interface for dataset management. All supported formats by Datumaro are also consumable by OTX without the need of explicit data conversion. If you want to build your own custom dataset format, you can do this via Datumaro CLI and API.

---

## Roadmap

### v1.0.0 (1Q23)

- Installation through PyPI
  - Package will be renamed as OTX (OpenVINO Training eXtension)
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
  - [OTX API](otx/api)
  - [OTX CLI](otx/cli)
  - [OTX Algorithms](otx/algorithms)
- Branches
  - [develop](https://github.com/openvinotoolkit/training_extensions/tree/develop)
    - Mainly maintained branch for releasing new features in the future
  - [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc)
    - Previously developed models can be found on this branch

---

# Quick start guide

In order to get started with OpenVINO™ Training Extensions see [the quick-start guide](QUICK_START_GUIDE.md).

---

# License

Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

---

## Issues / Discussions

Please use [Issues](https://github.com/openvinotoolkit/training_extensions/issues/new/choose) tab for your bug reporting, feature requesting, or any questions.

---

## Contributing

Please read the [Contribution guide](CONTRIBUTING.md) before starting work on a pull request.

---

## Known limitations

[misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch contains training, evaluation, and export scripts for models based on TensorFlow and PyTorch. These scripts are not ready for production. They are exploratory and have not been validated.

---
