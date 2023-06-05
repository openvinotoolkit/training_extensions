#!/bin/bash
DATASET_PREFIX="/home/yuchunli/_DATASET"
PKLOT="pklot-coco"
DOTA="dota_v1_coco"
OUTPUT_PATH="rotated_experiments"

# resnet50_maskrcnn ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/resnet50_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/resnet50_maskrcnn_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/resnet50_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/resnet50_maskrcnn_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/resnet50_maskrcnn_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/resnet50_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/resnet50_maskrcnn_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/resnet50_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/resnet50_maskrcnn_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/resnet50_maskrcnn_${PKLOT}
# ===================================================================================================================


# efficientnetb2b_maskrcnn ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/efficientnetb2b_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/efficientnetb2b_maskrcnn/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/efficientnetb2b_maskrcnn/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/efficientnetb2b_maskrcnn_${PKLOT}
# ===================================================================================================================

# rotated_atss_obb_mobilenet ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_mobilenet/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_mobilenet/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_mobilenet/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_mobilenet/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_mobilenet_${PKLOT}
# ===================================================================================================================

# rotated_atss_obb_r50 ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_r50/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_r50_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_r50/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/rotated_atss_obb_r50_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_r50_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_r50/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_r50_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_atss_obb_r50/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/rotated_atss_obb_r50_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_atss_obb_r50_${PKLOT}
# ===================================================================================================================

# rotated_fcos_mobilenet ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_mobilenet/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/rotated_fcos_mobilenet_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_mobilenet/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/rotated_fcos_mobilenet_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_fcos_mobilenet_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_mobilenet/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/rotated_fcos_mobilenet_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_mobilenet/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/rotated_fcos_mobilenet_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_fcos_mobilenet_${PKLOT}
# ===================================================================================================================

# rotated_fcos_r50 ===================================================================================================================
python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_r50/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${DOTA} \
--val-data-roots ${DATASET_PREFIX}/${DOTA} \
--workspace ${OUTPUT_PATH}/rotated_fcos_r50_${DOTA}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_r50/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${DOTA} \
--load-weights ${OUTPUT_PATH}/rotated_fcos_r50_${DOTA}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_fcos_r50_${DOTA}

python otx/cli/tools/train.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_r50/template.yaml \
--train-data-roots ${DATASET_PREFIX}/${PKLOT} \
--val-data-roots ${DATASET_PREFIX}/${PKLOT} \
--workspace ${OUTPUT_PATH}/rotated_fcos_r50_${PKLOT}

python otx/cli/tools/eval.py otx/algorithms/detection/configs/rotated_detection/rotated_fcos_r50/template.yaml \
--test-data-roots ${DATASET_PREFIX}/${PKLOT} \
--load-weights ${OUTPUT_PATH}/rotated_fcos_r50_${PKLOT}/outputs/latest_trained_model/models/weights.pth \
--workspace ${OUTPUT_PATH}/rotated_fcos_r50_${PKLOT}
# ===================================================================================================================