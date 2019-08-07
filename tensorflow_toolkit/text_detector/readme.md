# Text Detection in TensorFlow 2.0

This repository contains inference and training code for [PixelLink](https://arxiv.org/abs/1801.01315)-like model
networks. Models code is designed to enable export to frozen graph
and inference on CPU via OpenVINO. You can refer to [Original implementation](https://github.com/ZJULearning/pixel_link) as well.

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5.2
* TensorFlow 2.0
* CUDA 10.0

### Installation

It is recommended to use virtual environment, to do that please install `virtualenv`
```bash
sudo apt update
sudo apt install python3-dev python3-pip graphviz
sudo pip3 install -U virtualenv  # system-wide install

virtualenv -p python3 ./venv
source ./venv/bin/activate
```

To install required dependencies run

```bash
pip install -r requirements.txt
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
python annotation.py --type toy --images data --out_annotation data/annotation.json --imshow_delay 3
```

To create TFRecordDataset please run:

```bash
python create_dataset.py --input_datasets data/annotation.json --output data/annotation.tfrecord --imshow_delay 1
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
$ python train.py \
--learning_rate 0.001 \
--train_dir model \
--train_dataset ./data/annotation.tfrecord \
--batch_size 20 \
--epochs_per_evaluation 100 \
--test_dataset ./data/annotation.tfrecord
```

You can add one more parameter such as:
* `weights` - weights of pretrained model. That can give your faster convergence and better model.

```bash
$ python train.py \
--learning_rate 0.001 \
--train_dir model \
--train_dataset ./data/annotation.tfrecord \
--batch_size 20 \
--epochs_per_evaluation 100 \
--test_dataset ./data/annotation.tfrecord \
--weights some_pretrained_model.saved_weights
```


## Export TF-2.0 models to OpenVINO (IR)

* First step is to freeze your model. You will need configuration file that appears in `train_dir` during training.

```bash
python freeze.py \
--resolution 1280 768 \
--config configuration.yaml \
--weights some_pretrained_model.saved_weights
```

It will print information about frozen model and how to get IR:

```
Operations number: 51.075934092 GFlops

Output tensor names for using in InferenceEngine:
     model/link_logits_/add
     model/segm_logits/add
Run model_optimizer to get IR: mo.py --input_model /tmp/tmp4vhg5k2i/frozen_graph.pb --reverse_input_channels
```

* Second step is to run `mo.py`

You might need to install TF1.13 to run following line.

```bash
mo.py --input_model /tmp/tmp4vhg5k2i/frozen_graph.pb --reverse_input_channels
```

## Evaluation in TF-2.0

```bash
python test.py \
--config configuration.yaml \
--dataset dataset.tfrecord \
--weights some_pretrained_model.saved_weights
```

## Demo in OpenVINO

See https://github.com/opencv/open_model_zoo/tree/master/demos/text_detection_demo
