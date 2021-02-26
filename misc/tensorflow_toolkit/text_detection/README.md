# Text Detection in TensorFlow*

This repository contains inference and training code for [PixelLink](https://arxiv.org/abs/1801.01315)-like model
networks. Models code is designed to enable export to a frozen graph and inference on CPU via OpenVINO™.

> **NOTE**: Refer to the [original implementation](https://github.com/ZJULearning/pixel_link) for details.

[Trained models](https://download.01.org/opencv/openvino_training_extensions/models/text_detection/text_detection.tar.gz)

![](text-detection.jpg)

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5.2
* TensorFlow\* 2.0
* CUDA\* 10.0

### Installation

1. Create virtual environment:
    ```bash
    virtualenv venv -p python3 --prompt="(td)"
    ```

2. Activate virtual environment and set up OpenVINO™ variables:
    ```bash
    . venv/bin/activate
    ```

3. Install the module:
    ```bash
    pip3 install -e .
    ```

## <a name="Dataset"> Dataset </a>

### Sources

A toy dataset located in `./data`. You can use it to do all steps including the following:
* data preparation
* model training
* model evaluation

> **NOTE**: This dataset is considerably small. It is highly recommended to use at least several thousand images datasets like ICDAR-* (ICDAR 2013, ICDAR 2015, ICDAR 2017, ...), COCO-TEXT, MSRA-TD500, and others.

### Conversion

Use the `annotation.py` module to convert the datasets listed above to the internal format and create TFRecord that is suitable for training.

To convert a toy dataset located in the `./data` folder,run the command:

```bash
python tools/prepare_annotation.py \
  --type toy \
  --images data \
  --out_annotation data/annotation.json
```

To create TFRecordDataset, run the command:

```bash
python tools/create_dataset.py \
  --input_datasets data/annotation.json \
  --output data/annotation.tfrecord
```

## Training

To run training, run the following:
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
**Parameter description**
* `train_dir` - training directory where all snapshots and logs are stored
* `learning_rate` - estimation of how fast the model weighs are updated. Too high value might result in divergence, while too low value might be a reason of slow convergence.
* `train_dataset` - path to the TFRecord dataset that can be created using steps listed in the [dataset section](#Dataset) of this document. It is used during training.
* `batch_size` - batch size.
* `test_dataset` - path to the TFRecord dataset that can be created using steps listed in the [dataset section](#Dataset) of this document. It is used during validation to compute F1-score, precision, and recall.
* `epochs_per_evaluation` - the value showing how often the model is saved/evaluated

*Optional*:
* `weights` - weights of a pretrained model. Can increase convergence speed and result in a better model.

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

## Export models to OpenVINO™ (IR)

1. Freeze your model:
    > **NOTE**: Use the configuration file that appears in `train_dir` during training.

    ```bash
    python tools/export.py \
      --resolution 1280 768 \
      --config model/configuration.yaml \
      --weights model/weights/model-500.save_weights
    ```

    The command prints information about the frozen model and getting IR:

    ```
    Operations number: 51.075934092 GFlops

    Output tensor names for using in InferenceEngine:
        model/link_logits_/add
        model/segm_logits/add
    Run model_optimizer to get IR: mo.py --input_model model/weights/export/frozen_graph.pb --reverse_input_channels
    ```

2. Run the Model Optimizer.

    > **NOTE** You need to install TF1.13 to use the Model Optimizer.

    1. Create and activate new virtual environment:
    ```bash
    virtualenv venv_mo -p python3 --prompt="(td-mo)"
    . venv_mo/bin/activate
    ```

    3. Install modules and activate environment for OpenVINO™ :
    ```bash
    pip3 install -r requirements-mo.txt
    source /opt/intel/openvino/bin/setupvars.sh
    ```

    4. Run the Model Optimizer tool to export frozen graph to IR:
    ```bash
    mo.py --model_name text_detection \
      --input_model model/weights/export/frozen_graph.pb \
      --reverse_input_channels \
      --data_type FP32 \
      --input_shape="[1,768,1280,3]" \
      --output_dir IR
    ```

## Demo in OpenVINO™

See https://github.com/opencv/open_model_zoo/tree/master/demos/text_detection_demo.
