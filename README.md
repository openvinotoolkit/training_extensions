<div align="center">

# OpenVINO™ Training Extensions

---

[Key Features](#key-features) •
[Installation](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/installation.html) •
[Documentation](https://openvinotoolkit.github.io/training_extensions/latest/index.html) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/otx)](https://pypi.org/project/otx)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.1.1%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2024.0-purple)]()

<!-- markdownlint-enable  MD042 -->

[![Codecov](https://codecov.io/gh/openvinotoolkit/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/openvinotoolkit/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/openvinotoolkit/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/openvinotoolkit/training_extensions)
[![Pre-Merge Test](https://github.com/openvinotoolkit/training_extensions/actions/workflows/pre_merge.yaml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/pre_merge.yaml)
[![Build Docs](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yaml/badge.svg)](https://github.com/openvinotoolkit/training_extensions/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/otx?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/otx)

---

</div>

## Introduction

OpenVINO™ Training Extensions is a low-code transfer learning framework for Computer Vision.
The API & CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field.
OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on [PyTorch](https://pytorch.org) and [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

OpenVINO™ Training Extensions provides a "recipe" for every supported task type, which consolidates necessary information to build a model.
Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general.
If you are an experienced user, you can configure your own model based on [torchvision](https://pytorch.org/vision/stable/index.html), [mmcv](https://github.com/open-mmlab/mmcv) and [OpenVINO Model Zoo (OMZ)](https://github.com/openvinotoolkit/open_model_zoo).

Furthermore, OpenVINO™ Training Extensions provides automatic configuration for ease of use.
The framework will analyze your dataset and identify the most suitable model and figure out the best input size setting and other hyper-parameters.
The development team is continuously extending this [Auto-configuration](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/additional_features/auto_configuration.html) functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

### Key Features

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box support
- **Semantic segmentation**
- **Instance segmentation** including tiling algorithm support
- **Action recognition** including action classification and detection
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation
- **Visual Prompting** tasks including segment anything model, zero-shot visual prompting

OpenVINO™ Training Extensions supports the [following learning methods](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/algorithms/index.html):

- **Supervised**, incremental training, which includes class incremental scenario.

OpenVINO™ Training Extensions provides the following usability features:

- [Auto-configuration](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/additional_features/auto_configuration.html). OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model to provide the best accuracy/speed trade-off.
- [Datumaro](https://openvinotoolkit.github.io/datumaro/stable/index.html) data frontend: OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. We are constantly working to extend supported formats to give more freedom of datasets format choice.
- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Mixed-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient [hyper-parameter optimization module (HPO)](https://openvinotoolkit.github.io/training_extensions/latest/guide/explanation/additional_features/hpo.html). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.

---

## Installation

Please refer to the [installation guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/installation.html).
If you want to make changes to the library, then a local installation is recommended.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with otx.

```bash
pip install otx[base]
```

Alternatively, for zsh users:

```bash
pip install 'otx[base]'
```

</details>

<details>
<summary>Install from source</summary>
To install from source, you need to clone the repository and install the library using pip via editable mode.

```bash
# Use of virtual environment is highy recommended
# Using conda
yes | conda create -n otx_env python=3.10
conda activate otx_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/openvinotoolkit/training_extensions.git
cd training_extensions
pip install -e .[base]  # for zsh: pip install -e '.[base]'
```

</details>

---

## Quick-Start

OpenVINO™ Training Extensions supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use OpenVINO™ Training Extensions off-the-shelf.

For the CLI, the commands below provide subcommands, how to use each subcommand, and more:

```bash
# See available subcommands
otx --help

# Print help messages from the train subcommand
otx train --help

# Print help messages for more details
otx train --help -v   # Print required parameters
otx train --help -vv  # Print all configurable parameters
```

You can find details with examples in the [CLI Guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/cli_commands.html). and [API Quick-Guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

Below is how to train with auto-configuration, which is provided to users with datasets and tasks:

<details>
<summary>Training via API</summary>

```python
# Training with Auto-Configuration via Engine
from otx.engine import Engine

engine = Engine(data_root="data/wgisd", task="DETECTION")
engine.train()
```

For more examples, see documentation: [CLI Guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/cli_commands.html)

</details>

<details>
<summary>Training via CLI</summary>

```bash
otx train --data_root data/wgisd --task DETECTION
```

For more examples, see documentation: [API Quick-Guide](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/api_tutorial.html)

</details>

In addition to the examples above, please refer to the documentation for tutorials on using custom models, training parameter overrides, and [tutorial per task types](https://openvinotoolkit.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html), etc.

---

## Updates

### v2.1.0 (3Q24)

> _**NOTES**_
>
> OpenVINO™ Training Extensions, version 2.1.0 does not include the latest functional and security updates. OpenVINO™ Training Extensions, version 2.2.0 is targeted to be released in September 2024 and will include additional functional and security updates. Customers should update to the latest version as it becomes available.

### New features

- Add a flag to enable OV inference on dGPU
- Add early stopping with warmup. Remove mandatory background label in semantic segmentation task
- RTMDet-tiny enablement for detection task
- Add data_format validation and update in OTXDataModule
- Add torchvision.MaskRCNN
- Add Semi-SL for Multi-class Classification (EfficientNet-B0)
- Decoupling mmaction for action classification (MoviNet, X3D)
- Add Semi-SL Algorithms for mv3-large, effnet-v2, deit-tiny, dino-v2
- RTMDet-tiny enablement for detection task (export/optimize)
- Enable ruff & ruff-format into otx/algo/classification/backbones
- Add TV MaskRCNN Tile Recipe
- Add rotated det OV recipe

### Enhancements

- Change load_stat_dict to on_load_checkpoint
- Add try - except to keep running the remaining tests
- Update instance_segmentation.py to resolve conflict with 2.0.0
- Update XPU install
- Sync rgb order between torch and ov inference of action classification task
- Make Perf test available to load pervious Perf test to skip training stage
- Reenable e2e classification XAI tests
- Remove action detection task support
- Increase readability of pickling error log during HPO & fix minor bug
- Update RTMDet checkpoint url
- Refactor Torchvision Model for Classification Semi-SL
- Add coverage omit mm-related code
- Add docs semi-sl part
- Refactor docs design & Add contents
- Add execution example of auto batch size in docs
- Add Semi-SL for cls Benchmark Test
- Move value to device before logging for metric
- Add .codecov.yaml
- Update benchmark tool for otx2.1
- Collect pretrained weight binary files in one place
- Minimize compiled dependency files
- Update README & CODEOWNERS
- Update Engine's docstring & CLI --help outputs
- Align integration test to exportable code interface update for release branch
- Refactor exporter for anomaly task and fix a bug with exportable code
- Update pandas version constraint
- Include more models to export test into test_otx_e2e
- Move assigning tasks to Models from Engine to Anomaly Model Classes
- Refactoring detection modules

### Bug fixes

- Fix conflicts between develop and 2.0.0
- Fix polygon mask
- Fix vpm intg test error
- Fix anomaly
- Bug fix in Semantic Segmentation + enable DINOV2 export in ONNX
- Fix some export issues. Remove EXPORTABLE_CODE as export parameter.
- Fix `load_from_checkpoint` to apply original model's hparams
- Fix `load_from_checkpoint` args to apply original model's hparams
- Fix zero-shot `learn` for ov model
- Various fixes for XAI in 2.1
- Fix tests to work in a mm-free environment
- Fix a bug in benchmark code
- Update exportable code dependency & fix a bug
- Fix getting wrong shape during resizing
- Fix detection prediction outputs
- Fix RTMDet PTQ performance
- Fix segmentation fault on VPM PTQ
- Fix NNCF MaskRCNN-Eff accuracy drop
- Fix optimize with Semi-SL data pipeline
- Fix MaskRCNN SwinT NNCF Accuracy Drop

### Known issues

- Post-Training Quantization (PTQ) optimization applied to maskrcnn_swint in the instance segmentation task may result in significantly reduced accuracy. This issue is expected to be addressed with an upgrade to OpenVINO and NNCF in a future release.

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

- [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch contains training, evaluation, and export scripts for models based on TensorFlow and PyTorch.
  These scripts are not ready for production. They are exploratory and have not been validated.

---

## Disclaimer

Intel is committed to respecting human rights and avoiding complicity in human rights abuses.
See Intel's [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html).
Intel's products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

---

## Contributing

For those who would like to contribute to the library, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thank you! we appreciate your support!

<a href="https://github.com/openvinotoolkit/training_extensions/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/training_extensions" />
</a>

---
