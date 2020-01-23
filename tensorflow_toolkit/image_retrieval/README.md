# Image Retrieval

## Description

This code is intended to be used for image retrieval algorithm training when the probe image is transformed by the following transformation:
* Random cropping
* Rotation
* Repetition
* Color distortion

[Trained model](https://download.01.org/opencv/openvino_training_extensions/models/image_retrieval/image-retrieval-0001.tar.gz)

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.5 or 3.6
* TensorFlow\* 2.0.0a0 (for training only)
* OpenVINO™ 2019 R1 with Python API (to infer pretrained model only)

### Installation

1. Create virtual environment
    ```bash
    virtualenv venv -p python3 --prompt="(image_retrieval)"
    ```

2. Activate virtual environment and setup OpenVINO™ variables
    ```bash
    . venv/bin/activate
    ```

3. Install the modules
    ```
    pip install -e .
    ```

## Training

* To train an image-retrieval model, run the `train.py` script as follows:
  ```
  python tools/train.py \
    --gallery data/gallery/gallery.txt \
    --test_images data/queries/quieries.txt \
    --test_gallery data/gallery/gallery.txt \
    --train_dir model \
    --model mobilenet_v2 \
    --augmentation_config configs/augmentation_config.json \
    --loss triplet_1.0 \
    --steps_per_epoch 500 \
    --batch_size 32 \
    --input_size 224 \
    --dump_hard_examples \
    --lr_drop_step 500
  ```

* To fine-tune the image-retrieval model, run the `train.py` script as follows:
  ```
  python tools/train.py \
    --gallery data/gallery/gallery.txt \
    --test_images data/queries/quieries.txt \
    --test_gallery data/gallery/gallery.txt \
    --train_dir model \
    --model mobilenet_v2 \
    --augmentation_config configs/augmentation_config.json \
    --loss triplet_1.0 \
    --steps_per_epoch 500 \
    --batch_size 32 \
    --input_size 224 \
    --dump_hard_examples \
    --lr_drop_step 500 \
    --model_weights pretrained_model/weights-251920
  ```

The `augmentation_config.json` file contains the parameters of gallery images augmentation.

Each line in the file with list of gallery images should have the following format:
```
<path_to_image_folder>/<image_file_name> <id_of_gallery_group>
```
where `<id_of_gallery_group>` should be a number identifier to join similar (almost identical) gallery images
into groups (but in the simplest case, it can be different for each line).

## Evaluation

To test the image retrieval model using TensorFlow, run the `test.py` script as follows:
```
python tools/test.py \
  --model_weights pretrained_model/weights-251920 \
  --gallery data/gallery/gallery.txt \
  --test_images data/queries/quieries.txt \
  --ie tf
```

To test the image retrieval model using OpenVINO™, run the `test.py` script as follows:
```
python tools/test.py \
  --model_weights image-retrieval-0001.xml \
  --gallery data/gallery/gallery.txt \
  --test_images data/queries/quieries.txt \
  --ie ie \
  --cpu_extension /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx512.so
```

After running the command, you get the following:
```
 9	1.00	1.00	1.00	0.00
13	1.00	1.00	1.00	0.00
16	1.00	1.00	1.00	0.00
21	1.00	1.00	1.00	0.00
AVERAGE: top1: 1.000    top5: 1.000    top10: 1.000    mean_index: 0.000
AVERAGE top1 over all queries:  1.000
```

## Export

1. Freeze your model:

    ```
    python tools/export.py \
      --model mobilenet_v2 \
      --model_weights pretrained_model/weights-251920
    ```

2. Run the Model Optimizer:

    > **NOTE**: You need to install TF1.12 to use the Model Optimizer.

    1. Create and activate new virtual environment:
        ```bash
        virtualenv venv_mo -p python3 --prompt="(ir-mo)"
        . venv_mo/bin/activate
        ```

    2. Install modules and activate environment for OpenVINO™ :
        ```bash
        pip3 install -r requirements-mo.txt
        source /opt/intel/openvino/bin/setupvars.sh
        ```

    3. Run the Model Optimizer tool to export a frozen graph to Intermediate Representation:
        ```bash
        mo.py --model_name image-retrieval \
          --input_model model/export/frozen_graph.pb \
          --mean_values [127.5,127.5,127.5] \
          --scale 127.5 \
          --data_type FP32 \
          --output_dir model/export/IR
        ```
