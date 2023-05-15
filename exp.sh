#!/bin/bash
DATASET="Vitens-Coliform-coco"
OUTPUT_PATH="outputs_256_ir_scale_4_trial_2"

# otx train otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
# --train-data-roots /home/sungmanc/datasets/${DATASET} \
# --val-data-roots /home/sungmanc/datasets/${DATASET} \
# --workspace ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET} \
# params --tiling_parameters.enable_tiling 1 --tiling_parameters.tile_ir_scale_factor 1 --tiling_parameters.tile_sampling_ratio 0.1

# otx eval otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
# --test-data-roots /home/sungmanc/datasets/${DATASET} \
# --load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
# --output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}

# otx export otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
# --load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
# --output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/openvino

# otx eval otx/algorithms/detection/configs/instance_segmentation/resnet50_maskrcnn/template.yaml \
# --test-data-roots /home/sungmanc/datasets/${DATASET} \
# --load-weights ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}/openvino/openvino.xml \
# --output ${OUTPUT_PATH}/instance_segmentation_r50_maskrcnn_${DATASET}

# benchmark_app -m ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino/openvino.xml -api sync -niter 100 -nstreams 1 -hint none -nthreads 6

###
otx train otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--train-data-roots /home/sungmanc/datasets/${DATASET} \
--val-data-roots /home/sungmanc/datasets/${DATASET} \
--workspace ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET} \
params --tiling_parameters.enable_tiling 1 --tiling_parameters.tile_ir_scale_factor 4 --tiling_parameters.tile_sampling_ratio 0.1 \

otx eval otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots /home/sungmanc/datasets/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}

otx export otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/outputs/latest_trained_model/models/weights.pth \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino

otx eval otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots /home/sungmanc/datasets/${DATASET} \
--load-weights ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino/openvino.xml \
--output ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}

benchmark_app -m ${OUTPUT_PATH}/instance_segmentation_effnet_maskrcnn_${DATASET}/openvino/openvino.xml -api sync -niter 100 -nstreams 1 -hint none -nthreads 6