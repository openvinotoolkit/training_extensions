## Prepare datasets

### 1. Download datasets

To be able to train networks and/or get quality metrics for pre-trained ones,
it's necessary to download at least one dataset from following resources.
*  [ICDAR2013 (Focused Scene Text)](https://rrc.cvc.uab.es/?ch=2) - test part is used to get quality metric.
*  [ICDAR2015 (Incidental Scene Text)](https://rrc.cvc.uab.es/?ch=4)
*  [ICDAR2017 (MLT)](https://rrc.cvc.uab.es/?ch=8)
*  [ICDAR2019 (MLT)](https://rrc.cvc.uab.es/?ch=15)
*  [ICDAR2019 (ART)](https://rrc.cvc.uab.es/?ch=14)
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
python ./model_templates/horizontal-text-detection/tools/create_dataset.py \
    --config ./model_templates/horizontal-text-detection/tools/datasets/dataset_train.json \
    --output ${DATA_DIR}/text-dataset/IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json \
    --root ${DATA_DIR}/text-dataset/
export TRAIN_ANN_FILE=${DATA_DIR}/text-dataset/IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json
export TRAIN_IMG_ROOT=${DATA_DIR}/text-dataset
```
* Testing annotation
```bash
python ./model_templates/horizontal-text-detection/tools/create_dataset.py \
    --config ./model_templates/horizontal-text-detection/tools/datasets/dataset_test.json \
    --output ${DATA_DIR}/text-dataset/IC13TEST.json \
    --root ${DATA_DIR}/text-dataset/
export VAL_ANN_FILE=${DATA_DIR}/text-dataset/IC13TEST.json
export VAL_IMG_ROOT=${DATA_DIR}/text-dataset
export TEST_ANN_FILE=${VAL_ANN_FILE}
export TEST_IMG_ROOT=${VAL_IMG_ROOT}
```

Examples of json file for train and test dataset configuration can be found in `horizontal-text-detection/datasets`.
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

### 3. Using custom datasets

If you want to train/test network on your own dataset there are 3 ways to arrange the dataset preparing process:
1. If your dataset has COCO format, you may use its annotation right away.
2. You may convert your dataset to COCO format using [Datumaro](https://github.com/openvinotoolkit/datumaro) 
   or other tool.
3. The least preferable way is to modify converters already prepared in 
   [text_spotting.py](../../../../ote/ote/datasets/text_spotting.py) and use 
   [create_dataset.py](tools/create_dataset.py) script for conversion as it was explained above. 
   In that way please check that: 
   - `bbox` field has values in xywh format 
   - `segmentation` field contains at least 4 2d corner points (resulting in vector with 8 coordinates)
   - `transcription` field is not empty (if you don't intend to use transcription, put there every nonempty value like '0').
