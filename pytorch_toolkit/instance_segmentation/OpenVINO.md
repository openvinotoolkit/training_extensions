# Train Instance-Segmentation Models from Open Model Zoo

## instance-segmentation-security-0010

[Description](https://github.com/opencv/open_model_zoo/blob/develop/models/intel/instance-segmentation-security-0010/description/instance-segmentation-security-0010.md)

```bash
$ python3 tools/train.py \
    --dataset coco2017 \
    --max_image_size 800 1333 \
    --bs 16 \
    --bs_per_gpu 1 \
    --lr 0.02 \
    --max_iter 180000 \
    --drop_lr 120000 160000 \
    --model segmentoly.rcnn.model_zoo.resnet_panet_mask_rcnn.ResNeXt101PANetMaskRCNN \
    --load_backbone data/pretrained_models/converted/imagenet/detectron/resnext101.pth
```
> **NOTE**: This model is trained on 4 P100.

## instance-segmentation-security-0083

[Description](https://github.com/opencv/open_model_zoo/blob/develop/models/intel/instance-segmentation-security-0083/description/instance-segmentation-security-0083.md)

```bash
$ python3 tools/train.py \
    --dataset coco2017 \
    --max_image_size 480 640 \
    --bs 16 \
    --bs_per_gpu 4 \
    --lr 0.02 \
    --max_iter 90000 \
    --drop_lr 60000 80000 \
    --model segmentoly.rcnn.model_zoo.resnet_fpn_mask_rcnn.ResNeXt152FPNMaskRCNN \
    --load_backbone data/pretrained_models/converted/imagenet/detectron/resnext152.pth
```
> **NOTE**: This model is trained on 2 P100.

## instance-segmentation-security-0050

[Description](https://github.com/opencv/open_model_zoo/blob/develop/models/intel/instance-segmentation-security-0050/description/instance-segmentation-security-0050.md)

To train from ImageNet weights, run the command below:

```bash
$ python3 tools/train_0050.py
```

Find the fine-tuning sample in `tools/finetune_0050.py`.

> **NOTE**: Download checkpoints via `tools/download_pretrained_weights.py` first.
> Before running, make necessary changes to learning rate, batch size, number of training steps or
> other training parameters there.
