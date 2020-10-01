# Text Detection

Model that is able to detect more or less horizontal text with high speed on CPU.

| Model Name                  | Complexity (GFLOPs) | Size (Mp) | F1-score |    precision / recall   | Links                                                                                                                                    | GPU_NUM |
| --------------------------- | ------------------- | --------- | ------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| horizontal-text-detection-0001         | 7.72	            |  2.26     |  88.45% |    90.61% / 86.39%    | [model template](./horizontal-text-detection-0001/template.yaml), [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/horizontal-text-detection-0001.pth) | 2       |

## Training pipeline

### 0. Change a directory in your terminal to object_detection.

```bash
cd <training_extensions>/pytorch_toolkit/object_detection
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/horizontal-text-detection/horizontal-text-detection-0001/template.yaml`
export WORK_DIR=/tmp/my_model
python tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Download datasets

To be able to train networks and/or get quality metrics for pre-trained ones,  
it's necessary to download at least one dataset from following resources.  
*  [ICDAR2013 (Focused Scene Text)](https://rrc.cvc.uab.es/?ch=2) - test part is used to get quality metric.
*  [ICDAR2015 (Incidental Scene Text)](https://rrc.cvc.uab.es/?ch=4)
*  [ICDAR2017 (MLT)](https://rrc.cvc.uab.es/?ch=8)
*  [ICDAR2019 (MLT)](https://rrc.cvc.uab.es/?ch=15)
*  [ICDAR2019 (ART)](https://rrc.cvc.uab.es/?ch=14)
*  [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))   
*  [COCO-Text](https://bgshih.github.io/cocotext/)

### 3. Convert datasets

Extract downloaded datasets in `${DATA_DIR}/text-dataset` folder.

```bash
export DATA_DIR=${WORK_DIR}/data
```

Convert it to format that is used internally and split to the train and test part.

* Training annotation
```bash
python3 ./model_templates/horizontal-text-detection/tools/create_dataset.py \
    --config ./model_templates/horizontal-text-detection/datasets/dataset_train.json \
    --output ${DATA_DIR}/text-dataset/IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json
export TRAIN_ANN_FILE=${DATA_DIR}/text-dataset/IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json
export TRAIN_IMG_ROOT=${DATA_DIR}/text-dataset
```
* Testing annotation
```bash
python3 ./model_templates/horizontal-text-detection/tools/create_dataset.py \
    --config ./model_templates/horizontal-text-detection/datasets/dataset_test.json \
    --output ${DATA_DIR}/text-dataset/IC13TEST.json
export VAL_ANN_FILE=${DATA_DIR}/text-dataset/IC13TEST.json
export VAL_IMG_ROOT=${DATA_DIR}/text-dataset
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


### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training and Fine-tuning

Try both following variants and select the best one:

   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

   * If you would like to start **training** from pre-trained weights use `--load-weights` pararmeter.

      ```bash
      python train.py \
         --load-weights ${WORK_DIR}/snapshot.pth \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_IMG_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_IMG_ROOT} \
         --save-checkpoints-to ${WORK_DIR}/outputs
      ```

      Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

   * If you would like to start **fine-tuning** from pre-trained weights use `--resume-from` pararmeter and value of `--epochs` have to exceeds value stored inside `${MODEL_TEMPLATE}` file, otherwise training will be ended immideately. Here we add `5` additional epochs.

      ```bash
      export ADD_EPOCHS=5
      export EPOCHS_NUM=$((`cat ${MODEL_TEMPLATE} | grep epochs | tr -dc '0-9'` + ${ADD_EPOCHS}))

      python train.py \
         --resume-from ${WORK_DIR}/snapshot.pth \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_IMG_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_IMG_ROOT} \
         --save-checkpoints-to ${WORK_DIR}/outputs \
         --epochs ${EPOCHS_NUM}
      ```

### 6. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute MS-COCO metrics and save computed values to `${WORK_DIR}/metrics.yaml` run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

You can also save images with predicted bounding boxes using `--save-output-images-to` parameter.

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --save-output-images-to ${WORK_DIR}/output_images
```

### 7. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

For SSD networks an alternative OpenVINO™ representation is done automatically to `${WORK_DIR}/export/alt_ssd_export` folder.
SSD model exported in such way will produce a bit different results (non-significant in most cases),
but it also might be faster than the default one. As a rule SSD models in [Open Model Zoo](https://github.com/opencv/open_model_zoo/) are exported using this option.

### 8. Validation of IR

Instead of passing `snapshot.pth` you need to pass path to `model.bin` (or `model.xml`).

```bash
python eval.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```