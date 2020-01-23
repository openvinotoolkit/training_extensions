# Face Recognition in PyTorch*

## Introduction

*A repository for different experimental face-recognition (FR) models*

## Contents

1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train and Evaluation](#train-and-evaluation)
4. [Models](#models)
5. [Face Recognition Demo](#face-recognition-demo)
6. [Model Compression](#model-compression)
7. [Demo](#demo)

## Installation

Create and activate virtual python environment by running the command below:

```bash
cd $(git rev-parse --show-toplevel)/pytorch_toolkit/face_recognition
bash init_venv.sh
. venv/bin/activate
```


## Preparation

1. For a face-recognition training, download the [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) data. We will refer to this folder as `$VGGFace2_ROOT`.
2. For a face-recognition evaluation, download the [LFW](http://vis-www.cs.umass.edu/lfw/) data and [LFW landmarks](https://github.com/clcarwin/sphereface_pytorch/blob/master/data/lfw_landmark.txt).  Place everything in one folder, which will refer to as `$LFW_ROOT`.


## Train and Evaluate

1. To start training an FR model, run the command below:

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

2. To evaluate an FR snapshot, run the command as shown in the example below that uses MobileNet with the 256 embedding size trained for 300k:

```bash
python evaluate_lfw.py \
    --val_data_root $LFW_ROOT/lfw/ \
    --val_list $LFW_ROOT/pairs.txt \
    --val_landmarks $LFW_ROOT/lfw_landmark.txt \
    --snap /path/to/snapshot/mobilenet_256_300000.pt \
    --model mobilenet \
    --embed_size 256
```

## Configuration Files

Besides passing all the required parameters via the command line, the training script allows to read them from a `.yaml` configuration file.
Each line of such file should contain a valid description of one parameter in the YAML format.
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
Path to the configuration file can be passed to the training script via the command line. If any other arguments are passed before the configurations, they are overwritten
```bash
python train.py -m 0.35 @./my_config.yml #here m can be overwritten with the value from my_config.yml
```

## Models

1. You can download a pretrained model from fileshare as well:
 - [mobilenetv2](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_se_focal_121000.pt)
- [mobilenetv2_2x](https://download.01.org/opencv/openvino_training_extensions/models/face_recognition/Mobilenet_2x_se_121000.pt)

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

`mobilenetv2_2x` does not perform on the LFW benchmark as good as `mobilenetv2`, but it is heavier and achieves a higher score in the
uncleaned version of the [MegaFace](http://megaface.cs.washington.edu/participate/challenge.html) benchmark: 73.77% and 70.2% rank-1 at 1M distractors in reidentification protocol respectively.


## Model Compression

To train compressed models, use the [Neural Network Compression Framework (NNCF)](https://github.com/opencv/openvino_training_extensions/tree/develop/pytorch_toolkit/nncf), which compresses networks using quantization and sparsification algorithms.

**LandNet compression results on the NGD dataset**

| Algorithm          | RMSE  | Config path                                    |
| :----------------- | :---: | :--------------------------------------------- |
| Original           | 0.078 | -                                              |
| Quantization INT8  | 0.078 | configs/landnet/landnet_ngd_int8.json          |
| Sparsification 52% | 0.082 | configs/landnet/landnet_ngd_sparsity.json      |
| INT8 + Spars 52%   | 0.080 | configs/landnet/landnet_ngd_int8_sparsity.json |

**MobileFaceNet compression results on the VGG2Face dataset**

| Algorithm          | Accuracy | Config path                                            |
| :----------------- | :------: | :----------------------------------------------------- |
| Original           |  99.47   | -                                                      |
| Quantization INT8  |   99.5   | configs/mobilefacenet/mobilefacenet_vgg2_int8.json     |
| Sparsification 52% |   99.5   | configs/mobilefacenet/mobilefacenet_vgg2_sparsity.json |


### Landmark Model

1. Train

To start LandNet compression, run the following:

> **NOTE**: Define the `--snap_to_resume` argument.

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

During the first iterations of a quantization training, it is expected that loss increases dramatically due to initialization of new quantization layers.

2. Evaluate

To evaluate a compressed LandNet model, put the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (total memory is about 10Gb) in the ```$CelebA_ROOT``` folder and run the following command:

```bash
python evaluate_landmarks.py \
    --val_data_root $CelebA_ROOT/Img/img_celeba \
    --val_landmarks $CelebA_ROOT/Anno \
    --dataset celeb \
    --val_batch_size 128 \
    --snapshot <PATH_TO_SNAPSHOT>
    --compr_config <PATH_TO_COMPRESSION_CONFIG>
```

For evaluating, use the same compression configurations as for training.


### Face-Recognition Demo

1. Train

To start MobileFaceNet compression, run the following:

> **NOTE**: Define the `--snap_to_resume` argument in the configurations file.

```bash
python train.py @./configs/mobilefacenet/mobilefacenet_vgg2.yml \
    --compr_config <PATH_TO_COMPRESSION_CONFIG>
```

2. Evaluate

To evaluate a compressed MobileFaceNet model, run the command below:

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

To set up a demo, go to the [Face Recognition demo with the OpenVINO™ Toolkit](./demo/README.md)
