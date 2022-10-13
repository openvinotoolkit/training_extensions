
mode="val_batch_1"
#data="fish"
data="bccd"
#model="cspdarknet_yolox"
#model="mobilenetv2_atss"
model="mobilenetv2_ssd"

for trial in "1" "2" "3"
do
    ote train \
    ./external/model-preparation-algorithm/configs/detection/${model}_cls_incr/template.yaml \
    --train-ann-files=/local/sungmanc/datasets/${data}/annotations/instances_train_16_1.json \
    --train-data-roots=/local/sungmanc/datasets/${data}/images/train \
    --val-ann-files=/local/sungmanc/datasets/${data}/annotations/instances_val_100.json \
    --val-data-roots=/local/sungmanc/datasets/${data}/images/val \
    --save-model-to=./logs/${model}_${data}_${mode}_${trial}/results
done