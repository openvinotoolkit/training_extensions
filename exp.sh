#!/bin/bash
DATA_ROOT="/local/sungmanc/datasets"
OUTPUT_PATH="/local/sungmanc/scripts/cvs-106990/outputs/new_focal_90"

for seed in "1"
do
    # ### Peanut
    # otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    # --train-data-roots ${DATA_ROOT}/Peanut-dataset-COCO \
    # --val-data-roots ${DATA_ROOT}/Peanut-dataset-COCO \
    # --workspace ${OUTPUT_PATH}/peanut_${seed} \
    # --seed ${seed}

    # ### CVS-103334: Ball
    # otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    # --train-data-roots ${DATA_ROOT}/cvs-103334/Ball-dataset-COCO \
    # --val-data-roots ${DATA_ROOT}/cvs-103334/Ball-dataset-COCO \
    # --workspace ${OUTPUT_PATH}/ball_${seed} \
    # --seed ${seed}

    # ### CVS-106781: Logo
    # otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    # --train-data-roots ${DATA_ROOT}/cvs-106781/logo-dataset \
    # --val-data-roots ${DATA_ROOT}/cvs-106781/logo-dataset \
    # --workspace ${OUTPUT_PATH}/logo_${seed} \
    # --seed ${seed}

    # ### Fish: small
    # otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    # --train-data-roots ${DATA_ROOT}/fish-small \
    # --val-data-roots ${DATA_ROOT}/fish-small \
    # --workspace ${OUTPUT_PATH}/fish-small_${seed} \
    # --seed ${seed}

    # ### Vitens: small
    # otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    # --train-data-roots ${DATA_ROOT}/vitens-small \
    # --val-data-roots ${DATA_ROOT}/vitens-small \
    # --workspace ${OUTPUT_PATH}/vitens-small_${seed} \
    # --seed ${seed}

    ## Pothole: small
    otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
    --train-data-roots ${DATA_ROOT}/pothole-small \
    --val-data-roots ${DATA_ROOT}/pothole-small \
    --workspace ${OUTPUT_PATH}/pothole-small_${seed} \
    --seed ${seed}
done
# ### Fish
# otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
# --train-data-roots ${DATA_ROOT}/fish \
# --val-data-roots ${DATA_ROOT}/fish \
# --workspace ${OUTPUT_PATH}/fish

# ### Vitens
# otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
# --train-data-roots ${DATA_ROOT}/vitens \
# --val-data-roots ${DATA_ROOT}/vitens \
# --workspace ${OUTPUT_PATH}/vitens

# ### Pothole
# otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml \
# --train-data-roots ${DATA_ROOT}/pothole \
# --val-data-roots ${DATA_ROOT}/pothole \
# --workspace ${OUTPUT_PATH}/pothole
