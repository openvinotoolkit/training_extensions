# Text Detection in TensorFlow

This repository contains inference and training code for [PixelLink](https://arxiv.org/abs/1801.01315)-like model
networks. Models code is designed to enable export to frozen graph
and inference on CPU via OpenVINO. You can refer to [Original implementation](https://github.com/ZJULearning/pixel_link) as well.

Trained models: [link](https://download.01.org/opencv/openvino_training_extensions/models/text_detection/text_detection.tar.gz)

![](text-detection.jpg)

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* TensorFlow 2.0
* CUDA 10.0

### Installation

Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(td)"
```

2. Activate virtual environment and setup OpenVINO variables
```bash
. venv/bin/activate
```

3. Install the module
```bash
pip3 install -e .
```

## <a name="Dataset"> Dataset </a>

### Sources

There is a toy dataset located in `./data`. You can use it to do all steps including:
* data preparation
* model training
* model evaluation

But this is very very small dataset. It is highly recommended to use several thousand images at least.
You can find following datasets that can be used for training, evaluation, fine-tuning:
* ICDAR-* (ICDAR 2013, ICDAR 2015, ICDAR 2017, ...)
* COCO-TEXT
* MSRA-TD500
* ...

### Conversion

You can use `annotation.py` module to convert listed above datasets to internal format and that create TFRecord that is suitable for training.

To convert toy dataset located in `./data` folder please run:

```bash
python tools/annotation.py \
  --type toy \
  --images data \
  --out_annotation data/annotation.json
```

To create TFRecordDataset please run:

```bash
python tools/create_dataset.py \
  --input_datasets data/annotation.json \
  --output data/annotation.tfrecord
```

## Training

To run training you need to specify :
* `train_dir` - training directory where all snapshots and logs will be stored
* `learning_rate` - how fast model weighs are updated, too high value might be a reason of divergence, too low value might be a reason of slow convergence.
* `train_dataset` - path to TFRecord dataset that can be created using steps listed in [Dataset](#Dataset) section of this readme. It is using during training.
* `batch_size` - batch size.
* `test_dataset` - path to TFRecord dataset that can be created using steps listed in **Dataset** section of this readme. It is using during validation to compute F1-score, precision and recall.
* `epochs_per_evaluation` - how often model will be saved/evaluated.

```bash
python tools/train.py \
  --learning_rate 0.001 \
  --train_dir model \
  --train_dataset data/annotation.tfrecord \
  --epochs_per_evaluation 100 \
  --test_dataset data/annotation.tfrecord \
  --model_type mobilenet_v2_ext \
  --config configs/config.yaml
```

You can add one more parameter such as:
* `weights` - weights of pretrained model. That can give your faster convergence and better model.

```bash
python tools/train.py \
  --learning_rate 0.0001 \
  --train_dir model \
  --train_dataset data/annotation.tfrecord \
  --epochs_per_evaluation 100 \
  --test_dataset data/annotation.tfrecord \
  --model_type mobilenet_v2_ext \
  --config configs/config.yaml \
  --weights init_weights/model_mobilenet_v2_ext/weights/model-523.save_weights
```

## Evaluation

```bash
python tools/test.py \
  --config model/configuration.yaml \
  --dataset data/annotation.tfrecord \
  --weights model/weights/model-500.save_weights
```

## Export models to OpenVINO (IR)

* First step is to freeze your model. You will need configuration file that appears in `train_dir` during training.

```bash
python tools/export.py \
  --resolution 1280 768 \
  --config model/configuration.yaml \
  --weights model/weights/model-500.save_weights
```

It will print information about frozen model and how to get IR:

```
Operations number: 51.075934092 GFlops

Output tensor names for using in InferenceEngine:
     model/link_logits_/add
     model/segm_logits/add
Run model_optimizer to get IR: mo.py --input_model model/weights/export/frozen_graph.pb --reverse_input_channels
```

* Second step is to run `mo.py`

**NOTE** You need to install TF1.13 to use model optimizer.

1. Create and activate new virtual environment
```bash
virtualenv venv_mo -p python3 --prompt="(td-mo)"
. venv_mo/bin/activate
```

3. Install modules and activate environment for OpenVINO
```bash
pip3 install -r requirements-mo.txt
source /opt/intel/openvino/bin/setupvars.sh
```

4. Run model optimizer tool to export frozen graph to IR
```bash
mo.py --input_model model/weights/export/frozen_graph.pb --reverse_input_channels
```


## Demo in OpenVINO

See https://github.com/opencv/open_model_zoo/tree/master/demos/text_detection_demo
