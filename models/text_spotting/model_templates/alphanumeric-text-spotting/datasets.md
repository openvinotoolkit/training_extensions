## Prepare datasets

### 1. Download datasets

To be able to train networks and/or get quality metrics for pre-trained ones,
it's necessary to download at least one dataset from following resources.
*  [ICDAR2013 (Focused Scene Text)](https://rrc.cvc.uab.es/?ch=2) - test part is used to get quality metric.
*  [ICDAR2015 (Incidental Scene Text)](https://rrc.cvc.uab.es/?ch=4)
*  [ICDAR2017 (MLT)](https://rrc.cvc.uab.es/?ch=8)
*  [ICDAR2019 (MLT)](https://rrc.cvc.uab.es/?ch=15)
*  [ICDAR2019 (ART)](https://rrc.cvc.uab.es/?ch=14)
   * To exclude TotalText test set, please visit https://github.com/cs-chan/Total-Text-Dataset and use Total_Text_ID_vs_ArT_ID.list file during conversion (see ./model_templates/alphanumeric-text-spotting/dataset_train.json file).
*  [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
*  [COCO-Text](https://bgshih.github.io/cocotext/)

### 2. Convert datasets

Extract downloaded datasets in `${DATA_DIR}/text-dataset` folder.

```bash
export DATA_DIR=${WORK_DIR}/data
```

Convert it to format that is used internally and split to the train and test part.

* Training annotation
```bash
python3 ./tools/create_dataset.py \
    --config ./model_templates/alphanumeric-text-spotting/dataset_train.json \
    --output ${DATA_DIR}/text-dataset/dataset_train.json \
    --root ${DATA_DIR}/text-dataset/
export TRAIN_ANN_FILE=${DATA_DIR}/text-dataset/dataset_train.json
export TRAIN_IMG_ROOT=${DATA_DIR}/text-dataset
```
* Testing annotation
```bash
python3 ./tools/create_dataset.py \
    --config ./model_templates/alphanumeric-text-spotting/dataset_test.json \
    --output ${DATA_DIR}/text-dataset/dataset_val.json \
    --root ${DATA_DIR}/text-dataset/
export VAL_ANN_FILE=${DATA_DIR}/text-dataset/dataset_val.json
export VAL_IMG_ROOT=${DATA_DIR}/text-dataset
```

Examples of json file for train and test dataset configuration can be found in `alphanumeric-text-spotting/datasets`.
So, if you would like not to use all datasets above, please change its content.

The structure of the folder with datasets:
```
${DATA_DIR}/text-dataset
    ├── coco-text
    ├── icdar2013
    ├── icdar2015
    ├── icdar2017
    ├── icdar2019_art
    ├── icdar2019_mlt
    ├── MSRA-TD500
    ├── IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json
    └── IC13TEST.json
```
