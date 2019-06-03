# Action Recognition

This is the implementation of **Video Transformer Network** approach for Action Recognition in PyTorch. The repository also contains training code for other action recognition models, such as 3D CNNs, LSTMs, I3D, R(2+1)D, Two stream networks.

## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
5. [Action Recognition Demo](#demo)


## Requirements

The code is tested on Python 3.5, with dependencies listed in `requirements.txt` file. You can install required packages with:

```bash
pip install -r requirements.txt
```

You may also need to install FFmpeg in order to prepare training data:

```bash
sudo apt-get install ffmpeg
```

## Preparation

You need to download and pre-process Action Recognition dataset first:

### Getting the data

#### Kinetics
You can download [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset and split videos into 10 second clips using [these instructions](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/README.md).

Convert annotation files to json using provided python script:

```bash
python3 utils/kinetics_json.py ${data}/kinetics/kinetics-400_train.csv ${data}/kinetics/kinetics-400_val.csv ${data}/kinetics/kinetics-400_test.csv ${data}/kinetics/kientics_400.json
```

#### Mini-Kinetics

Download video list for subset of Kinetics [here](https://download.01.org/opencv/openvino_training_extensions/datasets/mini-kinetics/mini-kinetics-200.zip). You can follow the same instructions as for complete
Kinetics for data downloading and pre-processing.

#### UCF-101

Download UCF-101 and train-test split [here](http://crcv.ucf.edu/data/UCF101.php)

Convert all splits to json:

```bash
python3 utils/ucf101_json.py ${data}/ucf-101/ucfTrainTestlist
```

#### HMDB-51

HMDB-51 videos and train/test splits can be found [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

Convert all splits to json:

```bash
python3 utils/hmdb51_json.py ${data}/hmdb-51/splits/
```

### Converting videos
You may want to pre-process video files in order to speed up data loading and/or save some disk space.

You must convert videos either into *video* (.mp4) or *frames* (.jpg) format (controlled by `--video-format` option).
*Frames* format takes more disk space but significantly improves data loading performance,
however *video* format saves disk space, but takes more time for decoding.

You may also want to re-scale your videos to (128x or 256x), which is also saves disk space and improves data-loading performance.

Convert your videos, using the provided script. For example:

```bash
python3 utils/preprocess_videos.py --annotation_file ${data}/kinetics/kinetics_400.json \
    --raw_dir ${data}/kinetics/data \
    --destination_dir ${data}/kinetics/frames_data \
    --video-size 256 \
    --video-format frames \
    --threads 6
```

### Prepare configuration files

You need to create configuration file or update existing in `./datasets` directory
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

### Command line options
For complete list of options run `python3 main.py --help`. Here is the summary of some important options:

* `--result-path` -- Directory where logs and checkpoints will be stored. If you provide path to an directory from previous runs, the training will be resumed from latest checkpoint unless `--no-resume-train` is provided.
* `--model` -- Name of the model. The string before the first underscore symbol may be recognized as an encoder name (e.g. resnet34_vtn)en *ENCODER_DECODER*, you can find all implemented models at: `./action_recognition/models/`.
* `--clip-size` -- Number of frames in input clips. Note that you should multiply it by `--st` to get effective temporal receptive field
* `--st` -- Number of skipped frames, when sampling input clip. e.g. if st=2 then every 2nd frame will be skipped.
* `--resume-path` -- Path to checkpoint with pre-trained model, either for validation or fine-tuning.
* `--no-cuda` -- Use this option in environment without CUDA

### Examples
#### Validate trained model
```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --no-train --no-val --test --pretrain-path ~/resnet34_vtn.pth
```

#### Train model (with ImageNet pretrain)
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4
```

#### Continue training from checkpoint
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4 --resume-path ~/save_100.pth
```

#### Continue training from last checkpoint
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name/2 --dataset kinetics --model resnet34_vtn  \
    --batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4
```

#### Fine-tune pretrained model (e.g. from Kinetics to UCF)
```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset ucf101 --model resnet34_vtn \
    --batch 64 -j 12 --clip-size 16 --st 2 --lr 1e-5 --pretrain-path ~/resnet34_vtn_kinetcs.pth
```

#### Convert model to ONNX and OpenVINO format:

**NOTE** Modules that used `LayerNormalization` can be converted to ONNX only with flag `--no-layer-norm`,
  but this may lead to decrease the accuracy of converted model.
  Otherwise the script will crash with message `RuntimeError: ONNX export failed: Couldn't export operator aten::std`.

PyTorch to ONNX:
```bash
python3 main.py --model resnet34_vtn --clip-size 16 --st 2 --pretrain-path ~/resnet34_vtn_kinetics.pth --onnx resnet34_vtn.onnx
```

ONNX to OpenVINO:
```bash
mo.py --input_model resnet34_vtn.onnx --input_shape '[1,16,3,224,224]'
```

## Models
We provide some pre-trained models for your convenience:

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
You can try your models after converting it to the OpenVINO format or [pre-trained model from OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_Pre_Trained_Models.html#action_recognition_models) using the [demo application from OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_sample_action_recognition_README.html
