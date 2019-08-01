# Text Recognition in TensorFlow

This repository contains inference and training code for LSTM-based text recognition networks.
Models code is designed to enable export to frozen graph and inference on CPU via OpenVINO.

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* TensorFlow 1.13
* CUDA 10.0

### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(tr)"
```

2. Activate virtual environment and setup OpenVINO variables
```bash
. venv/bin/activate
. /opt/intel/openvino/bin/setupvars.sh
```
  **NOTE** Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
```
echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
```

3. Install the module
```bash
pip3 install -e .
```

## <a name="Dataset"> Dataset </a>

### Sources

There is a toy dataset located in `../../data/text_recognition/annotation.txt`. You can use it to do all steps including:
* model training
* model evaluation

But this is very very small dataset. It is highly recommended to use several thousand images at least.
You can find following datasets that can be used for training, evaluation, fine-tuning:
* [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)
* [SynthText in the Wild Dataset](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)

### Format

Your dataset shoud be stored in format of simple text file where each line has following format:
`<relative_path_to_image> <text>`

for example:

`./images/intel_1.jpg intel`

See `./data` for more details

## Training


```bash
python tools/train.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --learning_rate 0.1
```

You can add one more parameter such as:
* `weights_path` - weights of [pretrained model (checkpoint)](https://download.01.org/opencv/openvino_training_extensions/models/text_recognition/text_recognition.tar.gz). That can give your faster convergence and better model.

```bash
python tools/train.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --learning_rate 0.1 \
    --weights_path some_pre_trained_weights
```


## Export to OpenVINO (IR)

* First step is to freeze your model. The `freeze.py` script will save frozen graph to `./dump/graph.pb.frozen`.

```bash
python tools/export.py --weights_path some_pre_trained_weights
```

* Second step is to run `mo.py`

```bash
mo.py --input_model ./dump/graph.pb.frozen --framework tf
```

## Evaluation

```bash
python tools/test.py \
    --annotation_path ../../data/text_recognition/annotation.txt \
    --weights_path some_pre_trained_weights
```

## Demo in OpenVINO

See https://github.com/opencv/open_model_zoo/tree/master/demos/text_detection_demo
