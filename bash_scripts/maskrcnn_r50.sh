#!/bin/bash
DATASET_PREFIX="/home/yuchunli/datasets"
OUTPUT_PATH="/home/yuchunli/otx-v2/otx-workspace"

CONFIG="src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml"
FOLDER_NAME="maskrcnn_r50"

AEROMONAS="Vitens-Aeromonas-coco"
COLIFORM="Vitens-Coliform-coco"
CHICKEN="Chicken-Real-Time-coco-roboflow"
SKIN="skindetect-roboflow"
WGISD="wgisd-coco"
PCB="PCB_FICS_FPIC_1.v2i.coco-mmdetection"
BLUEBERRY="BlueBerry23.v1i.coco-mmdetection"
CARPART="car-seg.v1i.coco-mmdetection"
PACKAGE="factory_package.v1i.coco-mmdetection"

DATASET_ARRAY=(${COLIFORM} ${AEROMONAS}  ${CARPART} ${PACKAGE} ${CHICKEN} ${SKIN} ${WGISD} ${BLUEBERRY})

for dataset in ${DATASET_ARRAY[@]}; do
    mkdir -p ${OUTPUT_PATH}/${FOLDER_NAME}/$dataset
    otx train --config=${CONFIG} \
        --data_root=${DATASET_PREFIX}/$dataset \
        --seed=42 \
        --deterministic=False \
        --work_dir=${OUTPUT_PATH}/${FOLDER_NAME}/$dataset > ${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/train_raw.log

    sleep 2

    otx test --config="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/train/configs.yaml" \
        --checkpoint="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/train/best_checkpoint.ckpt" \
        --metric="otx.core.metrics.fmeasure.FMeasureCallable" > ${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/torch_test_raw.log

    sleep 2

    otx export --config="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/train/configs.yaml" \
        --checkpoint="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/train/best_checkpoint.ckpt"

    sleep 2

    otx test --config="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/train/configs.yaml" \
        --checkpoint="${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/.latest/export/exported_model.xml" \
        --metric="otx.core.metrics.fmeasure.FMeasureCallable" > ${OUTPUT_PATH}/${FOLDER_NAME}/$dataset/ov_test_raw.log
done
