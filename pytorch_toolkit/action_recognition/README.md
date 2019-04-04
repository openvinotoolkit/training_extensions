# Action Recognition

This is the implementation of **Video Transformer Network** approach for Action Recognition in PyTorch. The repository also contains training code for other action recognition models, such as 3D CNNs, LSTMs, I3D, R(2+1)D, Two stream networks. 

## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
5. [Action Recognition Demo](#demo)


## Requirements

The code is tested on Python 3, with dependencies listed in `requirements.txt` file. You can install required packages with:

```bash
pip intall -r requirements.txt
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

Download video list for subset of Kinetics [here](TBD). You can follow the same instructions as for complete
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

You can convert videos either into *video* (.mp4) or *frames* (.jpg) format. 
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

### Examples
#### Validate trained model
```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset kinetics --model resnet34_vtn
--batch 64 -j 12 --clip-size 16 --st 2 --no-train --no-val --test --pretrain-path ~/resnet34_vtn.pth
```

#### Train model (with ImageNet pretrain)
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn 
--batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4
```

#### Continue training from checkpoint
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name --dataset kinetics --model resnet34_vtn 
--batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4 --resume-path ~/save_100.pth
```

#### Continue training from last checkpoint
```bash
python3 main.py --root-path ~/data --result-path ~/logs/experiment_name/2 --dataset kinetics --model resnet34_vtn 
--batch 64 -j 12 --clip-size 16 --st 2 --epochs 120 --lr 1e-4 
```

#### Fine-tune pretrained model (e.g. from Kinetics to UCF)
```bash
python3 main.py --root-path ~/data --result-path ~/logs/ --dataset ucf101 --model resnet34_vtn 
--batch 64 -j 12 --clip-size 16 --st 2 --lr 1e-5 --pretrain-path ~/resnet34_vtn_kinetcs.pth
```

#### Convert model to ONNX and OpenVINO format:

PyTorch to ONNX:
```bash
python3 main.py --model resnet34_vtn --clip-size 16 --st 2 --pretrain-path ~/resnet34_vtn_kinetics.pth --onnx resnet34_vtn.onnx
```

ONNX to OpenVINO:
```bash
mo.py --input_model resnet34_vtn.onnx --input_shape '[1,16,3,224,224]'
```

## Models
We provide some pre-trained models in [Open Model Zoo](https://github.com/opencv/open_model_zoo)
TBD