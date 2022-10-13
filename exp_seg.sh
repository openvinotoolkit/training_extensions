#mode="val_batch_1"
mode="val_batch_8"
data="voc_person_car"

for model in "ocr-lite-hrnet-18" "ocr-lite-hrnet-18-mod2" "ocr-lite-hrnet-s-mod2"
do
    for trial in "1" "2" "3"
    do
        ote train \
        ./external/model-preparation-algorithm/configs/segmentation/${model}/template.yaml \
        --train-ann-files=/local/sungmanc/datasets/${data}/annotations/train \
        --train-data-roots=/local/sungmanc/datasets/${data}/images/train \
        --val-ann-files=/local/sungmanc/datasets/${data}/annotations/val \
        --val-data-roots=/local/sungmanc/datasets/${data}/images/val \
        --save-model-to=./logs/seg/${model}_${data}_${mode}_${trial}/results
    done
done