# Jester data preparation
Target datasets:
* [Jester](https://20bn.com/datasets/jester)

Pre-training datasets (to get the best performance):
* [ImageNet-1000](http://image-net.org/download) (2D backbone pre-training)
* [Kinetics-700](https://deepmind.com/research/open-source/kinetics) (full model pre-training)

> **Note**: To skip the pre-training stage we provide the [S3D MobileNet-V3](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-kinetics700.pth) pre-trained on both ImageNet-1000 and Kinetics-700 datasets.

## Data preparation

### 1. Download data

Register and download the [Jester](https://20bn.com/datasets/jester) data and follow the instructions to unpack it to `${DATA_DIR}/jester_data` folder.

```bash
export DATA_DIR=${WORK_DIR}/data
```

### 2. Convert dataset

Download [person-detection-asl-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/develop/models/intel/person-detection-asl-0001/description/person-detection-asl-0001.md) model from OMZ and copy it to `${WORK_DIR}`.
Crop frames by running the following command:

```bash
python3 ./tools/data/crop_images.py \
  -m ${WORK_DIR}/person-detection-asl-0001.xml \
  -i ${DATA_DIR}/jester_data/rawframes \
  -o ${DATA_DIR}/jester/global_crops
```

Convert annotation files by running the following commands:

```bash
python3 ./tools/data/prepare_jester_annot.py \
  -lm ${DATA_DIR}/jester_data/jester-v1-labels.csv \
  -im ${DATA_DIR}/jester/global_crops \
  -ia ${DATA_DIR}/jester_data/jester-v1-train.csv \
  -oa ${DATA_DIR}/jester/train.txt
python3 ./model_templates/gesture_recognition/tools/data/prepare_jester_annot.py \
  -lm ${DATA_DIR}/jester_data/jester-v1-labels.csv \
  -im ${DATA_DIR}/jester/global_crops \
  -ia ${DATA_DIR}/jester_data/jester-v1-validation.csv \
  -oa ${DATA_DIR}/jester/val.txt \
export TRAIN_ANN_FILE=train.txt
export TRAIN_DATA_ROOT=${DATA_DIR}
export VAL_ANN_FILE=val.txt
export VAL_DATA_ROOT=${DATA_DIR}
export TEST_ANN_FILE=val.txt
export TEST_DATA_ROOT=${DATA_DIR}
```

> **Note**: The labels for the test data split is not public, so we use the validation data split only to test a model internally.

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
   ├── jester
   |   ├── global_crops
   |   │   ├── 1
   |   |   |   ├── 00001.jpg
   |   │   |   └── ...
   |   |   └── ...
   |   ├── val.txt
   |   └── train.txt
   ├── imagenet
   |   └── train
   └── imagenet_train_list.txt
   ```
