# OpenVINO™ Training Extensions

[![python](https://img.shields.io/badge/python-3.8%2B-green)]()
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![mypy](https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat)]()
[![openvino](https://img.shields.io/badge/openvino-2022.3-purple)]()

> **_DISCLAIMERS_**: Some features described below are under development (refer to [feature/otx branch](https://github.com/openvinotoolkit/training_extensions/tree/feature/otx)). You can find more detailed estimation from the [Roadmap](#roadmap) section below.

## Overview

OpenVINO™ Training Extensions (OTE) is command-line interface (CLI) framework designed for low-code deep learning model training. OTE lets developers train/inference/optimize models with a diverse combination of model architectures and learning methods using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit). For example, users can train a ResNet18-based SSD ([Single Shot Detection](https://arxiv.org/abs/1512.02325)) model in a semi-supervised manner without worrying about setting a configuration manually. `ote build` and `ote train` commands will automatically analyze users' dataset and do necessary tasks for training the model with best configuration. OTE provides the following features:

- Provide a set of pre-configured models for quick start
  - `ote find` helps you quickly finds the best pre-configured models for common task types like classification, detection, segmentation, and anomaly analysis.
- Configure and train a model from torchvision, [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo)
  - `ote build` can help you configure your own model based on torchvision and OpenVINO Model Zoo models. You can replace backbones, necks and heads for your own preference (Currently only backbones are supported).
- Provide several learning methods including supervised, semi-supervised, imbalanced-learn, class-incremental, self-supervised representation learning
  - `ote build` helps you automatically identify the best learning methods for your data and model. All you need to do is to set your data in the supported format. If you don't specify a model, the framework will automatically sets the best model for you. For example, if your dataset has long-tailed and partially-annotated bounding box annotations, OTE auto-configurator will choose a semi-supervised imbalanced-learning method and an appropriate model with the best parameters.
- Integrated efficient hyper-parameter optimization
  - OTE has an integrated, efficient hyper-parameter optimization module. So, you don't need to worry about searching right hyper-parameters. Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- Support widely-used annotation formats
  - OTE uses [datumaro](https://github.com/openvinotoolkit/datumaro), which is designed for dataset building and transformation, as a default interface for dataset management. All supported formats by datumaro are also consumable by OTE without the need of explicit data conversion. If you want to build your own custom dataset format, you can do this via datumaro CLI and API.

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
  - [OTE SDK](ote_sdk)
  - [OTE CLI](ote_cli)
  - [OTE Algorithms](external)
- Branches
  - [develop](https://github.com/openvinotoolkit/training_extensions/tree/develop)
    - Mainly maintained branch for releasing new features in the future
  - [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc)
    - Previously developed models can be found on this branch

---

## Quick start guide

In order to get started with OpenVINO™ Training Extensions see [the quick-start guide](QUICK_START_GUIDE.md).

---

## License

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

Training, export, and evaluation scripts for TensorFlow- and most PyTorch-based models from the [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch are, currently, not production-ready. They serve exploratory purposes and are not validated.

---

\* Other names and brands may be claimed as the property of others.
