# Text Detection

Model that is able to detect more or less horizontal text with high speed on CPU.

| Model Name                  | Complexity (GFLOPs) | Size (Mp) | F1-score |    precision / recall   | Links                                                                                                                                    | GPU_NUM |
| --------------------------- | ------------------- | --------- | ------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| horizontal-text-detection-0001         | 7.72	            |  2.26     |  88.45% |    90.61% / 86.39%    | [configuration file](./horizontal-text-detection-0001/config.py), [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/horizontal-text-detection-0001.pth) | 2       |

## Training pipeline

### 0. Change a directory in your terminal to object_detection.

```bash
cd <openvino_training_extensions>/pytorch_toolkit/object_detection
```

### 1. Select a training configuration file and get pre-trained snapshot if available. Please see the table above.

```bash
export MODEL_NAME=horizontal-text-detection-0001
export CONFIGURATION_FILE==./horizontal-text-detection/$MODEL_NAME/config.py
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

Extract downloaded datasets in `data/text-dataset` folder.

Convert it to format that is used internally and split to the train and test part.

* Training annotation
```bash
python3 horizontal-text-detection/tools/create_dataset.py \
    --config horizontal-text-detection/datasets/dataset_train.json \
    --output data/text-dataset/IC13TRAIN_IC15_IC17_IC19_MSRATD500_COCOTEXT.json
```
* Testing annotation
```bash
python3 horizontal-text-detection/tools/create_dataset.py \
    --config horizontal-text-detection/datasets/dataset_test.json \
    --output data/text-dataset/IC13TEST.json
```

Examples of json file for train and test dataset configuration can be found in `horizontal-text-detection/datasets`.
So, if you would like not to use all datasets above, please change its content.

The structure of the folder with datasets:
```
object_detection/data/text-dataset
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

### 4. Training and Fine-tuning

Try both following variants and select the best one:

   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images.
   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

If you would like to start **training** from pre-trained weights do not forget to modify `load_from` path inside configuration file.

If you would like to start **fine-tuning** from pre-trained weights do not forget to modify `resume_from` path inside configuration file as well as increase `total_epochs`. Otherwise training will be ended immideately. If you would like to continue training with smaller learning rate, add the number of the resumed epoch to the `steps` field.

* To train the detector on a single GPU, run in your terminal:

   ```bash
   python3 ../../external/mmdetection/tools/train.py \
            $CONFIGURATION_FILE
   ```

* To train the detector on multiple GPUs, run in your terminal:

   ```bash
   ../../external/mmdetection/tools/dist_train.sh \
            $CONFIGURATION_FILE \
            <GPU_NUM>
   ```

* To train the detector on multiple GPUs and to perform quality metrics estimation as soon as training is finished, run in your terminal

   ```bash
   python horizontal-text-detection/tools/train_and_eval.py \
            $CONFIGURATION_FILE \
            <GPU_NUM>
   ```

### 5. Validation

* To dump detection of your model as well as compute metrics for text detection (F1-score, precision and recall) run:

   ```bash
   python ../../external/mmdetection/tools/test.py \
            $CONFIGURATION_FILE \
            <CHECKPOINT> \
            --out result.pkl \
            --eval f1
   ```
If you want to change the threshold for confidence of predictions which are used in calculations, change `score_thr` value in the `evaluation` section of the configuration file.
Tune it to increase precision (`score_thr` should be lower) or recall (`score_thr` should be higher). To compute MS-COCO metrics use `bbox` eval option instead of or with `f1`.

* You can also visualize the result of the detection. To do it use `result.pkl` obtained from previous step:

   ```bash
   python horizontal-text-detection/tools/visualize_text_detection.py \
            $CONFIGURATION_FILE \
            result.pkl
   ```
To visualize the dependence of recall from the instance size use `--draw_graph` option. To show predictions on the images use `--visualize` option.

### 6. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python ../../external/mmdetection/tools/export.py \
      $CONFIGURATION_FILE \
      <CHECKPOINT> \
      <EXPORT_FOLDER> \
      openvino
```

This produces model `$MODEL_NAME.xml` and weights `$MODEL_NAME.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

For SSD networks an alternative OpenVINO™ representation is possible.
To opt for it use extra `--alt_ssd_export` key to the `export.py` script.
SSD model exported in such way will produce a bit different results (non-significant in most cases),
but it also might be faster than the default one.

### 7. Validation of IR

Instead of running `test.py` you need to run `test_exported.py` and then repeat steps listed in [Validation paragraph](#5-validation).

```bash
python ../../external/mmdetection/tools/test_exported.py  \
      $CONFIGURATION_FILE \
      <EXPORT_FOLDER>/$MODEL_NAME.xml \
      --out results.pkl \
      --eval bbox
```

### 8. Demo

To see how the converted model works using OpenVINO you need to run `test_exported.py` with `--show` option.

```bash
python ../../external/mmdetection/tools/test_exported.py  \
      $CONFIGURATION_FILE \
      <EXPORT_FOLDER>/$MODEL_NAME.xml \
      --show
```

## Other

### Theoretical computational complexity estimation

To get per-layer computational complexity estimations, run the following command:

```bash
python ../../external/mmdetection/tools/get_flops.py \
       $CONFIGURATION_FILE
```
