#!/bin/bash
DATASET_PREFIX="/home/sungmanc/datasets"
#DATASET="Vitens-Coliform-coco"
DATASET="Vitens-Kiemgetal-coco-full"
OUTPUT_PATH="outputs_temp"

otx train otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DATASET} \
--val-data-roots ${DATASET_PREFIX}/${DATASET} \
--workspace ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET} \
params --tiling_parameters.enable_tiling 1 --tiling_parameters.tile_ir_scale_factor 2 --tiling_parameters.tile_sampling_ratio 0.1 \
--postprocessing.confidence_threshold 0.3 --learning_parameters.batch_size 12 \
--tiling_parameters.object_tile_ratio 0.06 --postprocessing.use_ellipse_shapes 1

otx eval otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}

otx export otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
--load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/openvino

otx eval otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/openvino/openvino.xml \
--output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}

<<<<<<< HEAD
# ###
=======
benchmark_app -m ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/openvino/openvino.xml -api sync -niter 100 -nstreams 1 -hint none -nthreads 6

###
otx train otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DATASET} \
--val-data-roots ${DATASET_PREFIX}/${DATASET} \
--workspace ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET} \
params --tiling_parameters.enable_tiling 1 --tiling_parameters.tile_ir_scale_factor 2 --tiling_parameters.tile_sampling_ratio 0.1 \
--postprocessing.confidence_threshold 0.3 --learning_parameters.batch_size 12 \
--tiling_parameters.object_tile_ratio 0.06 --postprocessing.use_ellipse_shapes 1

otx eval otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}

otx export otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino

otx eval otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino/openvino.xml \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}
