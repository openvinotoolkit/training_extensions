# Action Recognition

This is the implementation of the **Video Transformer Network** approach for Action Recognition in PyTorch\*. The repository also contains training code for other action-recognition models, such as 3D CNNs, LSTMs, I3D, R(2+1)D, and two-stream networks.

## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
5. [Action Recognition Demo](#demo)


## Requirements

The code is tested on Python\* 3.5, with dependencies listed in the `requirements.txt` file.
To install the required packages, run the following:

```bash
pip install -r requirements.txt
```

You might also need to install FFmpeg in order to prepare training data:

```bash
sudo apt-get install ffmpeg
```

## Preparation

Download and preprocess an Action Recognition dataset as described in the sections below.

### Get the Data

#### Kinetics

Download the [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset and split videos into 10-second clips using [these instructions](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/README.md).

Convert annotation files to the JSON format using the provided Python script:

```bash
python3 utils/kinetics_json.py ${data}/kinetics/kinetics-400_train.csv ${data}/kinetics/kinetics-400_val.csv ${data}/kinetics/kinetics-400_test.csv ${data}/kinetics/kientics_400.json
```

#### Mini-Kinetics

Download the [video list for subset of Kinetics](https://download.01.org/opencv/openvino_training_extensions/datasets/mini-kinetics/mini-kinetics-200.zip). You can follow the same instructions as for complete
Kinetics for data downloading and preprocessing.

#### UCF-101

Download the [UCF-101 and train-test split](https://crcv.ucf.edu/data/UCF101.php)

Convert all splits to the JSON format:

```bash
python3 utils/ucf101_json.py ${data}/ucf-101/ucfTrainTestlist
```

#### HMDB-51

Download the [HMDB-51 videos and train/test splits](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

Convert all splits to the JSON format:

```bash
python3 utils/hmdb51_json.py ${data}/hmdb-51/splits/
```

### Convert Videos

You can preprocess video files in order to speed up data loading and/or save some disk space.

Convert videos either into the *video* (.mp4) or the *frames* (.jpg) format (controlled by the `--video-format` option).
The *frames* format takes more disk space but significantly improves data loading performance,
while the *video* format saves disk space but takes more time for decoding.

Rescaling your videos to (128x or 256x) also saves disk space and improves data-loading performance.

Convert your videos using the provided script. For example:

```bash
python3 utils/preprocess_videos.py --annotation_file ${data}/kinetics/kinetics_400.json \
    --raw_dir ${data}/kinetics/data \
    --destination_dir ${data}/kinetics/frames_data \
    --video-size 256 \
    --video-format frames \
    --threads 6
```

### Prepare Configuration Files

You need to create a configuration file or update the existing one in the `./datasets` directory
for your dataset to adjust paths and other parameters.

The default structure of data directories is the following:

```misc
.../
    data/ (root dir)
        kinetics/
            frames_data/ (video path)
                .../ (directories of class names)
                    .../ (directories of video names)
                        ... (jpg files)
            kinetics_400.json (annotation path)
```

## Train/Eval

After you prepared the data, you can train or validate your model. Use commands below as an example.

### Command-Line Options

For complete list of options, run `python3 main.py --help`. The summary of some important options:

* `--result-path` -- Directory where logs and checkpoints are stored. If you provide the path to a directory from previous runs, the training is resumed from the latest checkpoint unless you provide `--no-resume-train`.
* `--model` -- Name of the model. The string before the first underscore symbol may be recognized as an encoder name (like resnet34_vtn)en *ENCODER_DECODER*. You can find all implemented models at: `./action_recognition/models/`.
* `--clip-size` -- Number of frames in input clips. Note that you should multiply it by `--st` to get effective temporal receptive field.
* `--st` -- Number of skipped frames when sampling input clip. For example, if st=2, every second frame is skipped.
* `--resume-path` -- Path to checkpoint with a pretrained model, either for validation or fine-tuning.
* `--no-cuda` -- Use this option in an environment without CUDA

### Examples

#### Validate a Trained Model

```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --no-train --no-val --test --pretrain-path ~/resnet34_vtn.pth
```

#### Train a Model (with ImageNet Pretrain)

```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --n-epochs 120 --lr 1e-4
```

#### Continue Training from a Checkpoint

```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --n-epochs 120 --lr 1e-4 --resume-path ~/save_100.pth
```

#### Continue Training from the Last Checkpoint

```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name/2 --dataset kinetics --model resnet34_vtn  \
    --batch 64 -j 12 --clip-size 16 --st 2 --n-epochs 120 --lr 1e-4
```

#### Fine-tune a Pretrained Model (for example, from Kinetics to UCF)

```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset ucf101 --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --lr 1e-5 --pretrain-path ~/resnet34_vtn_kinetcs.pth
```

#### Convert a Model to the ONNX\* and OpenVINO™ format

> **NOTE**: Modules that used `LayerNormalization` can be converted to ONNX\* only with the `--no-layer-norm` > flag, but this might decrease the accuracy of the converted model.
> Otherwise, the script crashes with the following message: `RuntimeError: ONNX export failed: Couldn't export operator aten::std`.

PyTorch to ONNX:
```bash
python3 main.py --model resnet34_vtn --clip-size 16 --st 2 --pretrain-path ~/resnet34_vtn_kinetics.pth --onnx resnet34_vtn.onnx
```

ONNX to OpenVINO™:
```bash
mo.py --input_model resnet34_vtn.onnx --input_shape '[1,16,3,224,224]'
```

## Pretrained Models

| Model| Input | Dataset | Video@1| Checkpoint  |  Command |
|---|---|---|---|---|---|
| MobileNetV2-VTN| RGB | Kinetics  | 62.51%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/mobilenetv2_vtn_rgb_kinetics.pth) | `python main.py --dataset kinetics --model mobilenetv2_vtn -b32 --lr 1e-4 --seq 16 --st 2`  |
| ResNet34-VTN| RGB | Kinetics  | 68.32%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/resnet_34_vtn_rgb_kinetics.pth) | `python main.py --dataset kinetics --model resnet34_vtn -b32 --lr 1e-4 --seq 16 --st 2`  |
| ResNet34-VTN| RGB-diff | Kinetics  | 67.31%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/resnet_34_vtn_rgbd_kinetics.pth) | `python main.py --dataset kinetics --model resnet34_vtn_rgbdiff -b32 --lr 1e-4 --seq 16 --st 2`  |
| SE-ResNext101_32x4d-VTN| RGB | Kinetics  | 69.52%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgb_kinetics.pth) | `python main.py --dataset kinetics --model se-resnext101-32x4d_vtn -b32 --lr 1e-4 --seq 16 --st 2 --no-mean-norm --no-std-norm` |
| SE-ResNext101_32x4d-VTN| RGB-diff | Kinetics  | 68.04%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgbd_kinetics.pth) | `python main.py --dataset kinetics --model se-resnext101-32x4d_vtn_rgbdiff -b32 --lr 1e-4 --seq 16 --st 2 --no-mean-norm --no-std-norm` |
| ResNet34-VTN| RGB | UCF101  | 90.27%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/resnet_34_vtn_rgb_ucf101_s1.pth) | `python main.py --dataset ucf101_1 --model resnet34_vtn -b16 --lr 1e-5 --seq 16 --st 2 --pretrain-path /PATH/TO/PRETRAINED/MODEL` |
| ResNet34-VTN| RGB-Diff | UCF101  | 93.02%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/resnet_34_vtn_rgbd_ucf101_s1.pth) | `python main.py --dataset ucf101_1 --model resnet34_vtn_rgbdiff -b16 --lr 1e-5 --seq 16 --st 2 --pretrain-path /PATH/TO/PRETRAINED/MODEL` |
| SE-ResNext101_32x4d-VTN| RGB | UCF101  | 91.8%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgb_ucf101_s1.pth) | `python main.py --dataset ucf101_1 --model se-resnext101-32x4d_vtn -b16 --lr 1e-5 --seq 16 --st 2 --no-mean-norm --no-std-norm --pretrain-path /PATH/TO/PRETRAINED/MODEL`  |
| SE-ResNext101_32x4d-VTN| RGB-diff | UCF101  | 93.44%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgbd_ucf101_s1.pth) | `python main.py --dataset ucf101_1 --model se-resnext101-32x4d_vtn_rgbdiff -b16 --lr 1e-5 --seq 16 --st 2 --no-mean-norm --no-std-norm --pretrain-path /PATH/TO/PRETRAINED/MODEL` |
| SE-ResNext101_32x4d-VTN| RGB | HMDB51  | 66.64%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgb_hmdb51_s1.pth) | `python main.py --dataset hmdb51_1 --model se-resnext101-32x4d_vtn -b16 --lr 1e-5 --seq 16 --st 2 --no-mean-norm --no-std-norm --pretrain-path /PATH/TO/PRETRAINED/MODEL` |
| SE-ResNext101_32x4d-VTN| RGB-diff | HMDB51  | 73.22%| [Download](https://download.01.org/opencv/openvino_training_extensions/models/action_recognition/se_resnext_101_32x4d_vtn_rgbd_hmdb51_s1.pth) | `python main.py --dataset hmdb51_1 --model se-resnext101-32x4d_vtn_rgbdiff -b16 --lr 1e-5 --seq 16 --st 2 --no-mean-norm --no-std-norm --pretrain-path /PATH/TO/PRETRAINED/MODEL`  |

## Demo

You can try your models after converting them to the OpenVINO™ format or a [pretrained model from OpenVINO™](https://docs.openvinotoolkit.org/latest/action_recognition_models.html) using the [demo application from OpenVINO™ toolkit](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_action_recognition_README.html)
