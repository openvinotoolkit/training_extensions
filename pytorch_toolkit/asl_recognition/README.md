# American Sign Language Recognition

This repository contains training scripts for American Sign Language (ASL) recognition neural network. The network is based on the [S3D](https://arxiv.org/abs/1712.04851) [MobileNetV3](https://arxiv.org/abs/1905.02244) architecture trained on the [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads) dataset. For inference is available 100 classes.
The code supports conversion to the ONNX\* format and inference of OpenVINO™ models.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* PyTorch\* 1.3.1
* OpenVINO™ 2019 R4 with Python API

### Installation

1. Create virtual environment and build mmaction:

   ```bash
   bash init_venv.sh
   ```

2. Activate virtual environment:

   ```bash
   . venv/bin/activate
   ```

## Datasets

Network was trained on the following datasets:

* ImageNet-1000 (2D pre-training)
* Kinetics-700 (3D pre-training)
* MS-ASL-1000 (final training)

## Data preparation

1. Download the [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads) annotation and unpack it to `msasl_data` folder.

2. Download MS-ASL videos using the annotation files (`MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`):

   ```bash
   python3 tools/download_msasl_videos.py \
     -s msasl_data/MSASL_train.json msasl_data/MSASL_val.json msasl_data/MSASL_test.json \
     -o msasl_data/videos
   ```

3. Extract frames and prepare annotation files by running the following command:

   ```bash
   python3 tools/extract_msasl_frames.py \
     -s msasl_data/MSASL_train.json msasl_data/MSASL_val.json msasl_data/MSASL_test.json \
     -v msasl_data/videos \
     -o ../../external/mmaction/data/msasl
   ```

4. Split annotation files by running the following commands:

   ```bash
   python3 tools/split_msasl_annotation.py \
     -a ../../external/mmaction/data/msasl/train.txt ../../external/mmaction/data/msasl/val.txt ../../external/mmaction/data/msasl/test.txt \
     -k 100 1000
   ```

5. To get the most robust model it's recomended to enable the [mixup](https://arxiv.org/abs/1710.09412) augmentation by specifying the paths to images in `imagenet_train_list.txt` file.
   In this repo we use ImageNet dataset but it's passible to use similar dataset with images. In case of other dataset you only need to create the `imagenet_train_list.txt` file with paths to images.
   If you have downloaded ImageNet dataset place it in `../../external/mmaction/data/imagenet` folder and extract image paths by running command:

   ```bash
   python3 tools/get_imagenet_paths.py \
     ../../external/mmaction/data/imagenet/train \
     ../../external/mmaction/data/imagenet_train_list.txt
   ```

6. Final `../../external/mmaction/data` directory should be like this:

   ```
   ../../external/mmaction/data
   ├── msasl
   |   ├── global_crops
   |   │   ├── video_name_0
   |   │   |   ├── clip_0000
   |   |   |   |   ├── img_00001.jpg
   |   |   |   |   └── ...
   |   │   |   └── ...
   |   |   └── ...
   |   ├── val100.txt
   |   ├── test100.txt
   |   └── train1000.txt
   ├── imagenet
   |   └── train
   └── imagenet_train_list.txt
   ```

> **Note**: Due to significant noise in the original annotation of MS-ASl dataset we use the cleanned version which includes:
>
> * Filtering invalid videos
> * Filtering invalid temporal crops
> * Enhancing temporal limits of gestures
> * Hiding text captions of presented gesture

## Training

1. Download [S3D-MobileNetV3 model](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-kinetics700.pth) pre-trained on Kinetics-700 dataset. Move the file with weights to the folder `../../external/mmaction/modelzoo`.

2. To train the model on a single GPU, run in your terminal:

    ```bash
    bash ../../external/mmaction/tools/dist_train_recognizer.sh \
      configs/s3d_rgb_mobilenetv3_stream.py \
      1 --validate \
      --data_dir ../../external/mmaction/data/ \
      --work_dir ../../external/mmaction/work_dirs/aslnet \
      --load_from ../../external/mmaction/modelzoo/s3d-mobilenetv3-large-statt-kinetics700.pth
    ```

   To change the batch size add the key `--num_videos <new_batch_size>` with new value (by default it's 14).

## Validation

1. To test the model on a single GPU, run in your terminal:

    ```bash
    python3 ../../external/mmaction/tools/test_recognizer.py \
      configs/s3d_rgb_mobilenetv3_stream.py \
      ../../external/mmaction/work_dirs/aslnet/epoch_<best_snapshot_id>.pth \
      --data_dir ../../external/mmaction/data/ \
      --mode test --num_classes 100 --gpus 1
    ```

## Conversion to the OpenVINO™ format

1. Convert PyTorch\* model to the ONNX\* format by running the script:

    ```bash
    python ../../external/mmaction/tools/onnx_export_recognizer.py \
      configs/s3d_rgb_mobilenetv3_stream.py \
      ../../external/mmaction/work_dirs/aslnet/epoch_<best_snapshot_id>.pth \
      ../../external/mmaction/work_dirs/asl-recognition.onnx \
      --check \
      --num_classes 100
    ```

2. Convert ONNX model to the OpenVINO™ format with the Model Optimizer with the command below:

    ```bash
    mo.py \
      --input_model ../../external/mmaction/work_dirs/asl-recognition.onnx  \
      --input input \
      --mean_values input[123.675,116.28,103.53] \
      --scale_values input[58.395,57.120,57.375] \
      --input_shape [1,3,16,224,224] \
      --output output \
      --model_name asl-recognition
    ```

  This produces model `asl-recognition.xml` and weights `asl-recognition.bin` in single-precision floating-point format (FP32). The obtained model expects image in planar RGB format.

## Estimate Theoretical Computational Complexity

To get computational complexity estimations, run the following command:

```bash
python3 ../../external/mmaction/tools/count_flops_recognizer.py \
  configs/s3d_rgb_mobilenetv3_stream.py
```

## Fine-Tuning

* Dataset should have the same data layout as MS-ASL format described in this instruction. Move the dataset to `../../external/mmaction/data/` folder.
* Change the model config file `configs/s3d_rgb_mobilenetv3_stream.py`:
  * Edit the `subdir_name` section of config to pass a custom name of dataset.
* Download the pre-trained on MS-ASL [model](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-msasl1000.pth). Move the file with weights to the folder `../../external/mmaction/modelzoo`.
* Fine-tune the network on the same way as the step 2 for training:

    ```bash
    bash ../../external/mmaction/tools/dist_train_recognizer.sh \
      configs/s3d_rgb_mobilenetv3_stream.py \
      1 --validate \
      --data_dir ../../external/mmaction/data/ \
      --work_dir ../../external/mmaction/work_dirs/aslnet \
      --load_from ../../external/mmaction/modelzoo/s3d-mobilenetv3-large-statt-msasl1000.pth
    ```

## OpenVINO™ Demo

OpenVINO™ provides the ASL Recognition demo, which is able to use the converted model. See details in the [demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/asl_recognition_demo).
