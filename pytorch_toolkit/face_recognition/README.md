# Face Recognition in PyTorch

## Introduction

*A repository for different experimental FR models.*

## Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
5. [Face Recognition Demo](#demo)

## Installation
1. Create and activate virtual python environment
```bash
bash init_venv.sh
. venv/bin/activate
```




## Preparation

1. For Face Recognition training you should download [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) data. We will refer to this folder as `$VGGFace2_ROOT`.
2. For Face Recognition evaluation you need to download [LFW](http://vis-www.cs.umass.edu/lfw/) data and [LFW landmarks](https://github.com/clcarwin/sphereface_pytorch/blob/master/data/lfw_landmark.txt).  Place everything in one folder, which will be `$LFW_ROOT`.




## Train/Eval
1. Go to `$FR_ROOT` folder
```bash
cd $FR_ROOT/
```

2. To start training FR model:
```bash
python train.py --train_data_root $VGGFace2_ROOT/train/ --train_list $VGGFace2_ROOT/meta/train_list.txt
--train_landmarks  $VGGFace2_ROOT/bb_landmark/ --val_data_root  $LFW_ROOT/lfw/ --val_list $LFW_ROOT/pairs.txt  
--val_landmarks $LFW_ROOT/lfw_landmark.txt --train_batch_size 200  --snap_prefix mobilenet_256 --lr 0.35
--embed_size 256 --model mobilenet --device 1
```

3. To evaluate FR snapshot (let's say we have MobileNet with 256 embedding size trained for 300k):
```bash
 python evaluate_lfw.py --val_data_root $LFW_ROOT/lfw/ --val_list $LFW_ROOT/pairs.txt
 --val_landmarks $LFW_ROOT/lfw_landmark.txt --snap /path/to/snapshot/mobilenet_256_300000.pt --model mobilenet --embed_size 256
```

## Configuration files
Besides passing all the required parameters via command line, the training script allows to read them from a `yaml` configuration file.
Each line of such file should contain a valid description of one parameter in the `yaml` fromat.
Example:
```yml
#optimizer parameters
lr: 0.4
train_batch_size: 256
#loss options
margin_type: cos
s: 30
m: 0.35
#model parameters
model: mobilenet
embed_size: 256
#misc
snap_prefix: MobileFaceNet
devices: [0, 1]
#datasets
train_dataset: vgg
train_data_root: $VGGFace2_ROOT/train/
#... and so on
```
Path to the config file can be passed to the training script via command line. In case if any other arguments were passed before the config, they will be overwritten.
```bash
python train.py -m 0.35 @./my_config.yml #here m can be overwritten with the value from my_config.yml
```



## Models

1. You can download pretrained model from fileshare as well - https://download.01.org/openvinotoolkit/open_model_zoo/training_toolbox_pytorch/models/fr/Mobilenet_se_focal_121000.pt
```bash
cd $FR_ROOT
python evaluate_lfw.py --val_data_root $LFW_ROOT/lfw/ --val_list $LFW_ROOT/pairs.txt --val_landmarks $LFW_ROOT/lfw_landmark.txt
--snap /path/to/snapshot/Mobilenet_se_focal_121000.pt --model mobilenet --embed_size 256
```

2. You should get the following output:
```
I1114 09:33:37.846870 10544 evaluate_lfw.py:242] Accuracy/Val_same_accuracy mean: 0.9923
I1114 09:33:37.847019 10544 evaluate_lfw.py:243] Accuracy/Val_diff_accuracy mean: 0.9970
I1114 09:33:37.847069 10544 evaluate_lfw.py:244] Accuracy/Val_accuracy mean: 0.9947
I1114 09:33:37.847179 10544 evaluate_lfw.py:245] Accuracy/Val_accuracy std dev: 0.0035
I1114 09:33:37.847229 10544 evaluate_lfw.py:246] AUC: 0.9995
I1114 09:33:37.847305 10544 evaluate_lfw.py:247] Estimated threshold: 0.7241
```

## Face Recognition Demo

1. For setting up demo, please go to [Face Recognition demo with OpenVINO Toolkit](./demo/README.md)
