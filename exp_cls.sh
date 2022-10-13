
for samples in "1000"
do
    data="cifar100"
    model="efficientnet_b0"
    ote train \
    ./external/model-preparation-algorithm/configs/classification/${model}_cls_incr/template.yaml \
    --train-ann-files=/local/sungmanc/datasets/cvs92405/${data}_${samples} \
    --train-data-roots=/local/sungmanc/datasets/cvs92405/${data}_${samples} \
    --val-ann-files=/local/sungmanc/datasets/${data}/test \
    --val-data-roots=/local/sungmanc/datasets/${data}/test \
    --save-model-to=./logs/${model}_${data}_${samples}/results
done