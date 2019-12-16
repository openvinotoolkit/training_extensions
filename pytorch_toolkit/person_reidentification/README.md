# Person re-identification

This repository contains training and inference code for person re-identification
neural network. The network is based on [OSNet](https://arxiv.org/abs/1905.00953)
architecture provided by [torchreid](https://github.com/KaiyangZhou/deep-person-reid.git)
module. The code supports conversion to ONNX format and inference OpenVINO models.

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* Pytorch 1.0 or higher
* OpenVINO 2019 R4 (or later) with Python API

### Installation

This repository needs `torchreid` module. To install it use the next command:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
git checkout 099b0ae7fcead522e56228860221a4f8b06cdaad
pip install -r requirements.txt
python setup.py develop
```

### Datasets

Networks were trained on the next datasets:

* Market-1501
* MSMT17v2

For training it is necessary to set up a root directory to datasets.
Structure of the root directory:

```
root
├── dukemtmc-reid
│   └── DukeMTMC-reID
│       ├── bounding_box_test
│       ├── bounding_box_train
│       └── query
│
├── market1501
│   └── Market-1501-v15.09.15
│       ├── bounding_box_test
│       ├── bounding_box_train
│       └── query
│
└── msmt17
    └── MSMT17_V2
        ├── mask_test_v2
        ├── mask_train_v2
        ├── list_gallery.txt
        ├── list_query.txt
        ├── list_train.txt
        └── list_val.txt
```

### Configuration file

Script for training and inference uses a configuration file.
[default_config.py](config/default_config.py) consists of default parameters.
This file also has description of parameters.
Parameters that you wish to change must be in your own configuration file.
Example: [person-reidentification-retail-0200.yaml](config/person-reidentification-retail-0200.yaml)

## Training

To start training create or choose configuration file and use the `main.py` script.
An example:

```bash
python main.py \
    --root /path/to/datasets/directory/root \
    --config config/person-reidentification-retail-0200.yaml
```

## Test

For test your network set in a configuration file parameter `test.evaluate` to `True`
and run a command like is used in training.
For visualization results set parameter `test.visrank` to True (it works only when
`test.evaluate` is `True`).
For visualization activation maps set parameter `test.visactmap` to True.


### Test OpenVINO models

To run OpenVINO model should have the one in IR format (*.xml and *.bin files).
To use this model just set in config file the next parameters:

```bash
model:
  openvino:
    name: /path/to/model/in/IR/format.xml
    cpu_extension: /path/to/cpu/extension/lib.so
```
*.xml and *.bin files should be saved in the same directory.


## Conversion pytorch model to ONNX format

To convert trained model from pytorch to ONNX format use the next command:

```bash
python convert_to_onnx.py \
    --config /path/to/config/file.yaml \
    --output-name /path/to/output/model \
    --verbose
```
Name of output model will be ended with `.onnx` automatically.
By default output model path is `model.onnx`. Be careful about parameter
`load_weights` in config file. `verbose` argument is non-required and
switches on detailed output in conversion function.
