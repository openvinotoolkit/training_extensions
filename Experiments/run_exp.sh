#!/bin/sh
# shellcheck disable=all

datasets=("ibean" "imagewoof2" "xray")
seeds=(0 1 2)
img_per_cls=20
models=("clip_base_16" "clip_base_32" "deit_base_16" "deit_small_16" "deit_tiny_16"
        "efficientnet_b0" "efficientnet_v2_s" "mobilenet_v3_large_1" "vit_base_16" "vit_base_32")

for seed in ${seeds[@]}
do
    for model in ${models[@]}
        do
            cd ${model}
            for dataset in ${datasets[@]}
                do
                    echo "Processing model and dataset: $model $dataset"
                    otx train --output outputs/${dataset}_${img_per_cls}_${seed} --data ../../data/${dataset}_${img_per_cls}_${seed}.yaml params \
                    --learning_parameters.num_iters 90 --learning_parameters.batch_size 8 --learning_parameters.learning_rate 0.001

                    rm -rf outputs/${dataset}_${img_per_cls}_${seed}/models \
                    rm -rf outputs/${dataset}_${img_per_cls}_${seed}/logs/*.pth
                done
            cd ..
        done
done
