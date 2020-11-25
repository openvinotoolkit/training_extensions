# Image classification

Models that are able to classify images on CPU.
                          | 4       |

## Datasets

## Training pipeline

### 0. Change a directory in your terminal to action_recognition_2.

```bash
cd <training_extensions>/pytorch_toolkit/image_classification
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/custom-classification/mobilenet_v2_w1/template.yaml`
export WORK_DIR=/tmp/my_model
python ../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Prepare data

### 3. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 4. Training and Fine-tuning

Try both following variants and select the best one:

* **Training from scratch** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
* **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

```bash
python train.py \
   --train-ann-files ${TRAIN_ANN_FILE} \
   --train-data-roots ${TRAIN_DATA_ROOT} \
   --val-ann-files ${VAL_ANN_FILE} \
   --val-data-roots ${VAL_DATA_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/outputs
```

> **NOTE**: It's recommended during fine-tuning to decrease the `--base-learning-rate` parameter compared with default value (see `${MODEL_TEMPLATE}`) to prevent from forgetting during the first iterations.

Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

### 5. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute mean accuracy metric run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 6. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar RGB format.
