# Text Recognition in TensorFlow*

This repository contains inference and training code for LSTM-based text recognition networks.
Models code is designed to enable export to frozen graph and inference on CPU via OpenVINO™.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5.2
* TensorFlow\* 1.13
* CUDA\* 10.0

### Installation

1. Create virtual environment:
    ```bash
    virtualenv venv -p python3 --prompt="(tr)"
    ```

2. Activate virtual environment and setup OpenVINO™ variables:
    ```bash
    . venv/bin/activate
    . /opt/intel/openvino/bin/setupvars.sh
    ```
    > **TIP**: Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
    ```
    echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
    ```

3. Install the module:
    ```bash
    pip3 install -e .
    pip3 install -e ../utils
    ```

## <a name="Dataset"> Dataset </a>

### Sources

A toy dataset located in `../../data/text_recognition/annotation.txt`. You can use it to do all steps including:
* model training
* model evaluation

> **NOTE**: This dataset is considerably small. It is highly recommended to use at least several thousand images datasets like [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).

### Format

Store your dataset in the format of a simple text file where each line has the following format:
`<relative_path_to_image> <text>`

Example:

`./images/intel_1.jpg intel`

See `./data` for more details.

## Training

```bash
python tools/train.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --learning_rate 0.1
```

Optional parameter:
* `weights_path` - weights of [pretrained model (checkpoint)](https://download.01.org/opencv/openvino_training_extensions/models/text_recognition/text_recognition.tar.gz). The parameter can increase convergence speed and result in a better model.

```bash
python tools/train.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --learning_rate 0.1 \
    --weights_path some_pre_trained_weights
```

## Export to OpenVINO™ Intermediate Representation (IR)

To run the model via OpenVINO™, freeze TensorFlow graph and then convert it to OpenVINO™ Intermediate Representation
(IR) using Model Optimizer:

```bash
python tools/export.py --checkpoint checkpoint_path \
    --data_type FP32 \
    --output_dir export
```

## Evaluation

```bash
python tools/test.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --weights_path some_pre_trained_weights
```

## Demo in OpenVINO™

See https://github.com/opencv/open_model_zoo/tree/master/demos/text_detection_demo.
