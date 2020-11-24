# Data preparation

This toolkit contains configs for training on the following datasets:

* [VeRi-776](https://github.com/JDAI-CV/VeRidataset)
* [VeRi-Wild](https://github.com/PKU-IMRE/VERI-Wild)
* UniverseModels (set of make/model classification datasets with merged annotation)
  - [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
  - [VMMRdb](https://github.com/faezetta/VMMRdb)

VeRi-776 and VeRi-Wild datasets are used without any changes and no future preparation are needed.

To get custom UniverseModels datasets several steps are needed to do:

1. Crop CompCars images according the original annotation:

    ```bash
    python ../../../external/deep-object-reid/tools/data/crop_compcars.py \
        --images-dir /path/to/source/compcars/image \
        --annot-dir /path/to/source/compcars/label \
        --output-dir /path/to/output/cropped_compcars/images
    ```

2. VMMRdb datsets is provided without any bounding box annotation so we need to align images manually. It's recommended to use HTC model ([config](https://github.com/open-mmlab/mmdetection/blob/master/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py), [weigts](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth)) from [mmdetection](https://github.com/open-mmlab/mmdetection).

    * Install [mmdet](https://github.com/openvinotoolkit/mmdetection/blob/ote/docs/install.md) package.
    * Dump vehicle masks by the command:
    ```bash
    python ../../../external/deep-object-reid/tools/data/dump_masks.py \
        --config /path/to/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py \
        --checkpoint /path/to/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth \
        --input_dir /path/to/source/vmmrdb/images \
        --output_dir /path/to/output/vmmrdb/masks
    ```
    * Crop images according vehicle masks by the command:
    ```bash
    python ../../../external/deep-object-reid/tools/data/align_images.py \
        --in-images /path/to/source/vmmrdb/images \
        --in-masks /path/to/output/vmmrdb/masks \
        --out-images /path/to/output/cropped_vmmrdb/images \
        --out-masks /path/to/output/cropped_vmmrdb/masks
    ```

3. Finally to get the UniverseModels dataset we need to merge the make/model/year annotation of both CompCars and VMMRdb datasets:

    ```bash
    python ../../../external/deep-object-reid/tools/data/merge_compcars_vmmrdb.py \
        --compcar-map /path/to/source/compcars/make_model_name.mat \
        --compcar-makes data/makes.txt \
        --compcar-images /path/to/output/cropped_compcars/images \
        --vmmrdb-images /path/to/output/cropped_vmmrdb/images \
        --output-dir /path/to/output/universemodels
    ```
