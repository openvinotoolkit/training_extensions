# OpenVINO™ Training Extensions
[![python](https://img.shields.io/badge/python-3.8%2B-green)]()
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![mypy](https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat)]()
[![openvino](https://img.shields.io/badge/openvino-2021.4-purple)]()

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.



## Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.8+
* [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) - for training on GPU

## Repository components
* [OTE SDK](ote_sdk)
* [OTE CLI](ote_cli)
* [OTE Algorithms](external)

## Quick start guide
In order to get started with OpenVINO™ Training Extensions see [the quick-start guide](QUICK_START_GUIDE.md).

## GitHub Repository
The project files can be found in [OpenVINO™ Training Extensions](https://github.com/openvinotoolkit/training_extensions).
Previously developed models can be found on the [misc branch](https://github.com/openvinotoolkit/training_extensions/tree/misc).

## License
Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Contributing
Please read the [Contribution guide](CONTRIBUTING.md) before starting work on a pull request.

## Known limitations
Training, export, and evaluation scripts for TensorFlow- and most PyTorch-based models from the [misc](#misc) branch are, currently, not production-ready. They serve exploratory purposes and are not validated.

---
\* Other names and brands may be claimed as the property of others.
