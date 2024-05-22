
# Instance-segmentation
## Training
otx train --config /home/harimkan/workspace/repo/otx-regression/src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml --data_root /home/harimkan/workspace/repo/otx-regression/data/CVPR_demo_datumaro_seed0 --work_dir otx-workspace/cli-ins-seg --seed 0 --data.config.data_format datumaro
## Evaluate
otx test --work_dir otx-workspace/cli-ins-seg
## Exporting
otx export --work_dir otx-workspace/cli-ins-seg
## XAI
otx explain --work_dir otx-workspace/cli-ins-seg --dump True --explain_config.postprocess True

# Detection
## Training
otx train --config /home/harimkan/workspace/repo/otx-regression/src/otx/recipe/detection/yolox_s.yaml --data_root /home/harimkan/workspace/repo/otx-regression/data/CVPR_demo_datumaro_seed0 --work_dir otx-workspace/cli-det --seed 0 --data.config.data_format datumaro
## Evaluate
otx test --work_dir otx-workspace/cli-det
## Exporting
otx export --work_dir otx-workspace/cli-det
## XAI
otx explain --work_dir otx-workspace/cli-det --dump True --explain_config.postprocess True

# Multi-Label Classification
## Training
otx train --config /home/harimkan/workspace/repo/otx-regression/src/otx/recipe/classification/multi_label_cls/efficientnet_b0.yaml --data_root /home/harimkan/workspace/repo/otx-regression/data/CVPR_demo_datumaro_seed0 --work_dir otx-workspace/cli-cls --seed 0 --data.config.data_format datumaro
## Evaluate
otx test --work_dir otx-workspace/cli-cls
## Exporting
otx export --work_dir otx-workspace/cli-cls
## XAI
otx explain --work_dir otx-workspace/cli-cls --dump True --explain_config.postprocess True
