# Integrating NNCF into MMLab mmdetection repository
https://github.com/open-mmlab/mmdetection

This folder contains a git patch to enable NNCF-based quantization-aware training for the object detection models on the COCO/Pascal VOC datasets from the *mmdetection* repository. Tested on SSD (with a VGG backbone) and RetinaNet (ResNet50 & ResNeXt101-64x4d backbones) models only (both quantization and quantization+sparsity pipelines).

Instructions:
1. Apply the `0001-Modifications-for-NNCF-usage.patch` file to the mmdetection repository checked out at commit id: `40ff3e5daf41522c553d3508385e55d1d83176ce`

2. To start quantization-aware fine-tuning of a model on the COCO dataset, you can use the regular configuration scripts provided in the repository for the specific model and dataset of interest. The only modification that is needed to be done is the addition of the `nncf_config` part to the full mmdetection config script. Examples of such configs are provided within the supplied patch (`configs/pascal_voc/ssd300_voc_int8.py` and `configs/retinanet_r50_fpn_1x_int8.py` files).
Make sure to download the uncompressed model checkpoint files to start from using the links specified in each .py config file.
Model fine-tuning can be run using the common mmdetection command lines. For instance, the following command line, while run from the mmdetection repository root, will launch SSD-VGG quantization fine-tuning on the VOC dataset (provided you set proper paths to the dataset in the config file):
`python tools/train.py configs/pascal_voc/ssd300_voc_int8.py`

    Distributed multiprocessing is also supported, simply use the corresponding version of the command line in the mmdetection repository:
`./tools/dist_train.sh configs/pascal_voc/ssd300_voc_int8.py ${GPU_NUMBER}`

    Note that in all cases the training hyperparameters might have to be adjusted to accomodate the hardware you have available.

# Current best results:

**RetinaNet-ResNet50-FPN**:

_Full-precision FP32 baseline model_ - 35.6 average box mAP on the `coco_2017_val` dataset.

_INT8 model (symmetrically quantized)_ - 35.3 average box mAP on the `coco_2017_val` dataset.

_INT8+sparse model (symmetrically quantized, 50% sparsity rate)_ - 34.7 average box mAP on the `coco_2017_val` dataset.

**RetinaNet-ResNeXt101-64x4d-FPN**:

_Full-precision FP32 baseline model_ - 39.6 average box mAP on the `coco_2017_val` dataset.

_INT8 model (symmetrically quantized)_ - 39.1 average box mAP on the `coco_2017_val` dataset.
