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

## Updates - v2.2.0 (3Q24)

### New features

- Add RT-DETR model for Object Detection
- Add Multi-Label & H-label Classification with torchvision models
- Add Hugging-Face Model Wrapper for Classification
- Add LoRA finetuning capability for ViT Architectures
- Add Hugging-Face Model Wrapper for Object Detection
- Add Hugging-Face Model Wrapper for Semantic Segmentation
- Enable torch.compile to work with classification
- Add `otx benchmark` subcommand
- Add RTMPose for Keypoint Detection Task
- Add Semi-SL MeanTeacher algorithm for Semantic Segmentation
- Update head and h-label format for hierarchical label classification
- Support configurable input size

### Enhancements

- Reimplement of ViT Architecture following TIMM
- Enable to override data configurations
- Enable to use input_size at transforms in recipe
- Enable to use polygon and bitmap mask as prompt inputs for zero-shot learning
- Refactoring `ConvModule` by removing `conv_cfg`, `norm_cfg`, and `act_cfg`
- Support ImageFromBytes
- enable model export
- Move templates from OTX1.X to OTX2.X
- Include Geti arrow dataset subset names
- Include full image with anno in case there's no tile in tile dataset
- Add type checker in converter for callable functions (optimizer, scheduler)
- Change sematic segmentation to consider bbox only annotations
- Relieve memory usage criteria on batch size 2 during adaptive batch size
- Remove background label from RT Info for segmentation task
- Prevent using too low confidence thresholds in detection

### Bug fixes

- Fix Combined Dataloader & unlabeled warmup loss in Semi-SL
- Revert #3579 to fix issues with replacing coco_instance with a different format in some dataset
- Add num_devices in Engine for multi-gpu training
- Add missing tile recipes and various tile recipe changes
- Change categories mapping logic
- Fix config converter for tiling
- Fix num_trials calculation on dataset length less than num_class
- Fix out_features in HierarchicalCBAMClsHead
- Fix multilabel_accuracy of MixedHLabelAccuracy
- Fix wrong indices setting in HLabelInfo

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
