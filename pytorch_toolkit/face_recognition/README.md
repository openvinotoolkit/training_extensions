# Face Recognition in PyTorch

## Introduction

*A repository for different experimental FR models.*

## Contents

1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train and evaluation](#train-and-evaluation)
4. [Models](#models)
5. [Face Recognition Demo](#face-recognition-demo)
6. [Model compression](#model-compression)
7. [Demo](#demo)

## Installation

1. Create and activate virtual python environment

```bash
cd $(git rev-parse --show-toplevel)/pytorch_toolkit/face_recognition
bash init_venv.sh
. venv/bin/activate
```


## Preparation

1. For Face Recognition training you should download [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) data. We will refer to this folder as `$VGGFace2_ROOT`.
2. For Face Recognition evaluation you need to download [LFW](http://vis-www.cs.umass.edu/lfw/) data and [LFW landmarks](https://github.com/clcarwin/sphereface_pytorch/blob/master/data/lfw_landmark.txt).  Place everything in one folder, which will be `$LFW_ROOT`.


## Train and evaluation

1. To start training FR model:

```bash
python train.py \
    --train_data_root $VGGFace2_ROOT/train/ \
    --train_list $VGGFace2_ROOT/meta/train_list.txt \
    --train_landmarks $VGGFace2_ROOT/bb_landmark/ \
    --val_data_root $LFW_ROOT/lfw/ \
    --val_list $LFW_ROOT/pairs.txt \
    --val_landmarks $LFW_ROOT/lfw_landmark.txt \
    --train_batch_size 200 \
    --snap_prefix mobilenet_256 \
    --lr 0.35 \
    --embed_size 256 \
    --model mobilenetv2 \
    --device 1
```

2. To evaluate FR snapshot (let's say we have MobileNet with 256 embedding size trained for 300k):

```bash
python evaluate_lfw.py \
    --val_data_root $LFW_ROOT/lfw/ \
    --val_list $LFW_ROOT/pairs.txt \
    --val_landmarks $LFW_ROOT/lfw_landmark.txt \
    --snap /path/to/snapshot/mobilenet_256_300000.pt \
    --model mobilenet \
    --embed_size 256
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
model: mobilenetv2
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

1. You can download pretrained model from fileshare as well - [mobilenetv2](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_se_focal_121000.pt),
[mobilenetv2_2x](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_2x_se_121000.pt).

```bash
python evaluate_lfw.py \
    --val_data_root $LFW_ROOT/lfw/ \
    --val_list $LFW_ROOT/pairs.txt \
    --val_landmarks $LFW_ROOT/lfw_landmark.txt \
    --snap /path/to/snapshot/Mobilenet_se_focal_121000.pt \
    --model mobilenet \
    --embed_size 256
```

2. You should get the following output:

- for `mobilenetv2`:

```
I1114 09:33:37.846870 10544 evaluate_lfw.py:242] Accuracy/Val_same_accuracy mean: 0.9923
I1114 09:33:37.847019 10544 evaluate_lfw.py:243] Accuracy/Val_diff_accuracy mean: 0.9970
I1114 09:33:37.847069 10544 evaluate_lfw.py:244] Accuracy/Val_accuracy mean: 0.9947
I1114 09:33:37.847179 10544 evaluate_lfw.py:245] Accuracy/Val_accuracy std dev: 0.0035
I1114 09:33:37.847229 10544 evaluate_lfw.py:246] AUC: 0.9995
I1114 09:33:37.847305 10544 evaluate_lfw.py:247] Estimated threshold: 0.7241
```

- for `mobilenetv2_2x`:
```
I0820 15:48:06.307454 23328 evaluate_lfw.py:262] Accuracy/Val_same_accuracy mean: 0.9893
I0820 15:48:06.307612 23328 evaluate_lfw.py:263] Accuracy/Val_diff_accuracy mean: 0.9990
I0820 15:48:06.307647 23328 evaluate_lfw.py:264] Accuracy/Val_accuracy mean: 0.9942
I0820 15:48:06.307732 23328 evaluate_lfw.py:265] Accuracy/Val_accuracy std dev: 0.0061
I0820 15:48:06.307766 23328 evaluate_lfw.py:266] AUC: 0.9992
I0820 15:48:06.307812 23328 evaluate_lfw.py:267] Estimated threshold: 0.6721
```

`mobilenetv2_2x` is slightly worse on the LFW benchmark than `mobilenetv2`, but it's heavier and achieves higher score in the
uncleaned version of the [MegaFace](http://megaface.cs.washington.edu/participate/challenge.html) benchmark: 73.77% rank-1 at 1M distractors in reidentification protocol vs 70.2%.


## Model compression

To train compressed models used
[NNCF](https://github.com/opencv/openvino_training_extensions/tree/develop/pytorch_toolkit/nncf), it is a framework for neural network compression using quantization and sparsification algorithms.

- **Landnet compression results on NGD dataset**

| Algorithm          | RMSE  | Config path                                    |
| :----------------- | :---: | :--------------------------------------------- |
| Original           | 0.078 | -                                              |
| Quantization int8  | 0.078 | configs/landnet/landnet_ngd_int8.json          |
| Sparsification 52% | 0.082 | configs/landnet/landnet_ngd_sparsity.json      |
| Int8 + Spars 52%   | 0.080 | configs/landnet/landnet_ngd_int8_sparsity.json |

- **MobileFaceNet compression results on VGG2Face dataset**

| Algorithm          | Accuracy | Config path                                            |
| :----------------- | :------: | :----------------------------------------------------- |
| Original           |  99.47   | -                                                      |
| Quantization int8  |   99.5   | configs/mobilefacenet/mobilefacenet_vgg2_int8.json     |
| Sparsification 52% |   99.5   | configs/mobilefacenet/mobilefacenet_vgg2_sparsity.json |


### Landmark model

1. Train

To start Landnet compression (remember that it is necessary to define '--snap_to_resume'):

```bash
python train_landmarks.py \
    --train_data_root $NDG_ROOT/mnt/big_ssd/landmarks_datasets \
    --train_landmarks $NDG_ROOT/list_train_large.json \
    --lr 0.4 \
    --train_batch_size 512 \
    --snap_prefix Landnet_Compr_ \
    --dataset ngd \
    --val_step 500 \
    --epoch_total_num 300 \
    --snap_to_resume <PATH_TO_SNAPSHOT> \
    --snap_folder snapshots/compression --compr_config <PATH_TO_COMPRESSION_CONFIG>
```

During the first iterations of a quantization training, it is expected that loss will increase dramatically. It is connected with initialization of new quantization layers.

2. Evaluate

To evaluate compressed Landnet put [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (total memory is about 10Gb) in folder, which will be ```$CelebA_ROOT``` . After that run the following command:

```bash
python evaluate_landmarks.py \
    --val_data_root $CelebA_ROOT/Img/img_celeba \
    --val_landmarks $CelebA_ROOT/Anno \
    --dataset celeb \
    --val_batch_size 128 \
    --snapshot <PATH_TO_SNAPSHOT>
    --compr_config <PATH_TO_COMPRESSION_CONFIG>
```

For evaluating use the same compression config as for training.


### Face recognition model

1. Train

To start MobileFaceNet compression (remember that it is necessary to define '--snap_to_resume' in configuration file):

```bash
python train.py @./configs/mobilefacenet/mobilefacenet_vgg2.yml \
    --compr_config <PATH_TO_COMPRESSION_CONFIG>
```

2. Evaluate

To evaluate compressed MobileFaceNet:

```bash
python evaluate_lfw.py \
    --val_data_root $LFW_ROOT/lfw/ \
    --val_list $LFW_ROOT/pairs.txt \
    --val_landmarks $LFW_ROOT/lfw_landmark.txt \
    --snap <PATH_TO_SNAPSHOT> \
    --model mobilenet \
    --embed_size 256 --compr_config <PATH_TO_COMPRESSION_CONFIG>
```


## Demo

1. For setting up demo, please go to [Face Recognition demo with OpenVINO Toolkit](./demo/README.md)
