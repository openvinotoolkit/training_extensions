#!/bin/bash
# seed 1
CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1_12.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1_12

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1_12/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1_12/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1_24.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1_12/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1_24

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1_24/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1_24/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1_36.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1_24/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1_36

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1_36/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1_36/eval_test.json

# seed 10
CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed10_12.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed10_12

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed10_12/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed10_12/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed10_24.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed10_12/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed10_24

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed10_24/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed10_24/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed10_36.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed10_24/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed10_36

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed10_36/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed10_36/eval_test.json

# seed 100
CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed100_12.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed100_12

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed100_12/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed100_12/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed100_24.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed100_12/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed100_24

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed100_24/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed100_24/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed100_36.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed100_24/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed100_36

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed100_36/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed100_36/eval_test.json


# seed 1234
CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1234_12.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1234_12

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1234_12/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1234_12/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1234_24.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1234_12/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1234_24

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1234_24/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1234_24/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed1234_36.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1234_24/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed1234_36

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed1234_36/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed1234_36/eval_test.json


# seed 4321
CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed4321_12.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed4321_12

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed4321_12/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed4321_12/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed4321_24.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed4321_12/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed4321_24

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed4321_24/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed4321_24/eval_test.json

CUDA_VISIBLE_DEVICES=0 ote train ../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--train-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_train_seed4321_36.json \
--train-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/train \
--val-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_val.json \
--val-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/val \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed4321_24/weights.pth \
--save-model-to logs/counting_exp/vitens_Enterococci/ote_seed4321_36

CUDA_VISIBLE_DEVICES=0 ote eval \
../external/mmdetection/configs/custom-counting-instance-seg/efficientnetb2b_maskrcnn/template_experimental.yaml \
--test-ann-files ../vitens_dataset/Vitens-Enterococci-coco/annotations/instances_test.json \
--test-data-roots ../vitens_dataset/Vitens-Enterococci-coco/images/test \
--load-weights logs/counting_exp/vitens_Enterococci/ote_seed4321_36/weights.pth \
--save-performance logs/counting_exp/vitens_Enterococci/ote_seed4321_36/eval_test.json
