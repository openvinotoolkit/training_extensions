# MS-ASL data preparation

Target datasets:
* [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads)

> **Note**: Due to significant noise in the original annotation of MS-ASl dataset we use the cleaned version which includes:
>
> * Filtering invalid videos
> * Filtering invalid temporal crops
> * Enhancing temporal limits of gestures
> * Hiding text captions of presented gesture

Pre-training datasets (to get the best performance):
* [ImageNet-1000](http://image-net.org/download) (2D backbone pre-training)
* [Kinetics-700](https://deepmind.com/research/open-source/kinetics) (full model pre-training)

> **Note**: To skip the pre-training stage we provide the [S3D MobileNet-V3](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-kinetics700.pth) pre-trained on both ImageNet-1000 and Kinetics-700 datasets.

## Data preparation

### 1. Download annotation

Download the [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads) annotation and unpack it to `${DATA_DIR}/msasl_data` folder.

```bash
export DATA_DIR=${WORK_DIR}/data
```

### 2. Download videos

Download MS-ASL videos using the unpacked annotation files (`MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`):

```bash
python3 ./tools/data/download_msasl_videos.py \
  -s ${DATA_DIR}/msasl_data/MSASL_train.json ${DATA_DIR}/msasl_data/MSASL_val.json ${DATA_DIR}/msasl_data/MSASL_test.json \
  -o ${DATA_DIR}/msasl_data/videos
```

### 3. Convert dataset

Extract frames and prepare annotation files by running the following command:

```bash
python3 ./tools/data/extract_msasl_frames.py \
  -s ${DATA_DIR}/msasl_data/MSASL_train.json ${DATA_DIR}/msasl_data/MSASL_val.json ${DATA_DIR}/msasl_data/MSASL_test.json \
  -v ${DATA_DIR}/msasl_data/videos \
  -o ${DATA_DIR}/msasl
```

Split annotation files by running the following commands:

```bash
python3 ./tools/data/split_msasl_annotation.py \
  -a ${DATA_DIR}/msasl/train.txt ${DATA_DIR}/msasl/val.txt ${DATA_DIR}/msasl/test.txt \
  -k 100
export TRAIN_ANN_FILE=train.txt
export TRAIN_DATA_ROOT=${DATA_DIR}
export VAL_ANN_FILE=val.txt
export VAL_DATA_ROOT=${DATA_DIR}
export TEST_ANN_FILE=test.txt
export TEST_DATA_ROOT=${DATA_DIR}
```

To get the most robust model it's recommended to enable the [mixup](https://arxiv.org/abs/1710.09412) augmentation by specifying the paths to images in `imagenet_train_list.txt` file.
Additionally you should enable MixUp by uncommenting appropriate line in `model.py` config.

In this repo we use ImageNet dataset but it's possible to use similar dataset with images. In case of other dataset you only need to create the `imagenet_train_list.txt` file with paths to images.
If you have downloaded ImageNet dataset place it in `${DATA_DIR}/imagenet` folder and dump image paths by running command:

```bash
python3 ./tools/data/get_imagenet_paths.py \
  ${DATA_DIR}/train \
  ${DATA_DIR}/imagenet_train_list.txt
```

Finally, the `${DATA_DIR}` directory should be like this:

   ```
   ${DATA_DIR}
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
