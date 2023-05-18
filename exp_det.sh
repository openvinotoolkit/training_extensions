#!/bin/bash
DATASET_PREFIX="/home/sungmanc/datasets"
# DATASET="Vitens-Coliform-coco"
DATASET="Vitens-Kiemgetal-coco-full"
OUTPUT_PATH="outputs_ATSS_mobilenet"

otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DATASET} \
--val-data-roots ${DATASET_PREFIX}/${DATASET} \
--workspace ${OUTPUT_PATH}/detection_ATSS_${DATASET} \
params --tiling_parameters.enable_tiling 1

otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/detection_ATSS_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/detection_ATSS_${DATASET}

otx export otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
--load-weights ${OUTPUT_PATH}/detection_ATSS_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/detection_ATSS_${DATASET}/openvino

otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/detection_ATSS_${DATASET}/openvino/openvino.xml \
--output ${OUTPUT_PATH}/detection_ATSS_${DATASET}/openvino