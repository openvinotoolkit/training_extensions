# Person re-identification

This repository contains training and inference code for person re-identification
neural networks. The networks are based on [OSNet](https://arxiv.org/abs/1905.00953)
architecture provided by [torchreid](https://github.com/KaiyangZhou/deep-person-reid.git)
project. The code supports conversion to ONNX format and inference of OpenVINO models.

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* PyTorch 1.3 or higher
* OpenVINO 2019 R4 (or later) with Python API

### Installation

1. Create and activate virtual python environment

```bash
cd $(git rev-parse --show-toplevel)/pytorch_toolkit/person_reidentification
bash init_venv.sh
. venv/bin/activate
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

### Configuration files

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

### Pretrained models

You can download pretrained models in PyTorch format corresponding to the provided configs from fileshare as well:
- [person-reidentification-retail-0103](link)
- [person-reidentification-retail-0107](link)
- [person-reidentification-retail-0200](link)


### Test OpenVINO reidentification models

OpenVINO models are represented by \*.xml and \*.bin files (IR format).
To use such a model just set in config file the next parameters:

```yaml
model:
  openvino:
    name: /path/to/model/in/IR/format.xml
    cpu_extension: /path/to/cpu/extension/lib.so
```
\*.xml and \*.bin files should be saved in the same directory.


## Conversion PyTorch model to OpenVINO format

The conversion is done in two stages: first - convert a PyTorch model to the ONNX format and second - convert the obtained ONNX model to the IR format.
To convert trained model from PyTorch to ONNX format use the next command:

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

After the ONNX model is obtained one can convert it to IR.
This produces model `model.xml` and weights `model.bin` in single-precision floating-point format (FP32):

```bash
python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model model.onnx  \
    --mean_values '[123.675, 116.28 , 103.53]' \
    --scale_values '[58.395, 57.12 , 57.375]' \
    --reverse_input_channels
```

## OpenVINO demo

OpenVINO provides multi-camera-multi-person tracking demo, which is able to use these models as person reidentification networks. Please, see details in the [demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/multi_camera_multi_person_tracking).
