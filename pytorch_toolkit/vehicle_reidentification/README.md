# Vehicle Re-Identification

This repository contains training and inference code for vehicle re-identification neural networks. The networks are based on the [OSNet](https://arxiv.org/abs/1905.00953) architecture provided by the [deep-object-reid](https://github.com/opencv/deep-object-reid.git) project. The code supports conversion to the ONNX\* format.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5.2
* PyTorch\* 1.3 or higher
* OpenVINO™ 2019 R4 (or lower) with Python API

### Installation

1. Create and activate virtual Python environment

```bash
cd $(git rev-parse --show-toplevel)/pytorch_toolkit/vehicle_reidentification
bash init_venv.sh
. venv/bin/activate
```

### Datasets

This tollkit contains configs for training on the following datasets:

* [VeRi-776](https://github.com/JDAI-CV/VeRidataset)
* [VeRi-Wild](https://github.com/PKU-IMRE/VERI-Wild)
* UniverseModels (set of make/model classification datasets with merged annotation)
  - [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
  - [VMMRdb](https://github.com/faezetta/VMMRdb)

> **Note**: The instruction how to prepare the training dataset can be found in [DATA.md](DATA.md)

The final Structure of the root directory is as follows:

```
root
├── veri
│   ├── image_train
│   ├── image_query
│   ├── image_test
│   └── train_label.xml
│
├── veri-wild
│   ├── images
│   └── vehicle_info.txt
│
└── universe_models
    └── images
```

### Configuration Files

The script for training and inference uses a configuration file
[default_config.py](https://github.com/opencv/deep-object-reid/blob/ote/scripts/default_config.py), which consists of default parameters.
This file also has description of parameters.
Parameters that you wish to change must be in your own configuration file.
Example: [vehicle-reid-0001.yaml](configs/vehicle-reid-0001.yaml)

## Training

To start training, create or choose a configuration file and use the [main.py](https://github.com/opencv/deep-object-reid/blob/ote/scripts/main.py) script.

Example:

```bash
python ../../external/deep-object-reid/scripts/main.py \
    --root /path/to/datasets/directory/root \
    --config configs/vehicle-reid-0001.yaml
```

## Fine-tuning

To start fine-tuning, create or choose a configuration file, choose init model weigts (you can use the pre-trained weights -- see [section](#pre-trained-models)) and use the [main.py](../../external/deep-object-reid/scripts/main.py) script.

Example:

```bash
python ../../external/deep-object-reid/scripts/main.py \
    --root /path/to/datasets/directory/root \
    --config configs/vehicle-reid-0001.yaml \
    model.load_weights /path/to/pretrained/model/weigts
```

## Testing

To test your network, specify your configuration file and use the [main.py](../../external/deep-object-reid/scripts/main.py) script.

Example:

```bash
python ../../external/deep-object-reid/scripts/main.py \
    --root /path/to/datasets/directory/root \
    --config configs/vehicle-reid-0001.yaml \
    model.load_weights /path/to/trained/model/weigts \
    test.evaluate True
```

## Pre-trained Models

You can download pre-trained models in the PyTorch format corresponding to the provided configs from a fileshare as well:
- [vehicle-reid-0001](https://drive.google.com/open?id=1KaDrzX3xZFwO986Q0a06U10beM-s_eBZ)


## Convert a PyTorch Model to the OpenVINO™ Format

Follow the steps below:

1. Convert a PyTorch model to the ONNX format by running the following:

    ```bash
    python ../../external/deep-object-reid/tools/convert_to_onnx.py \
        --config /path/to/config/file.yaml \
        --output-name /path/to/output/model \
        model.load_weights /path/to/trained/model/weigts
    ```

    Name of the output model ends with `.onnx` automatically.
    By default, the output model path is `model.onnx`.

2. Convert the obtained ONNX model to the IR format by running the following command:

```bash
python <OpenVINO_INSTALL_DIR>/deployment_tools/model_optimizer/mo.py --input_model model.onnx  \
    --input_shape [1,3,208,208]
    --reverse_input_channels
```

This produces the `model.xml` model and weights `model.bin` in single-precision floating-point format (FP32).

## OpenVINO™ Demo

OpenVINO™ provides the multi-camera-multi-target tracking demo, which is able to use these models as vehicle re-identification networks. See details in the [demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/multi_camera_multi_target_tracking).

## Citation

Original repository: [github.com/sovrasov/deep-person-reid/tree/vehicle_reid](https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid)
