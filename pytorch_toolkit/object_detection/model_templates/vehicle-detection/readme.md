# Vehicle Detection

Models that are able to detect vehicles.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- |
| vehicle-detection-0200 | 0.82 | 1.83 | 25.4 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-detection-0200-1.pth), [model template](./vehicle-detection-0200/template.yaml) | 4 |
| vehicle-detection-0201 | 1.84 | 1.83 | 32.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-detection-0201-1.pth), [model template](./vehicle-detection-0201/template.yaml) | 4 |
| vehicle-detection-0202 | 3.28 | 1.83 | 36.9 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-detection-0202-1.pth), [model template](./vehicle-detection-0202/template.yaml) | 4 |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 0. Change a directory in your terminal to object_detection.

```bash
cd <training_extensions>/pytorch_toolkit/object_detection
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/vehicle-detection/vehicle-detection-0200/template.yaml`
export WORK_DIR=/tmp/my_model
python tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Collect dataset

Collect or download images with vehicles presented on them. One can download MS-COCO dataset and remain images with cars only.
```bash
export DATA_DIR=${WORK_DIR}/data
wget http://images.cocodataset.org/zips/val2017.zip -P ${DATA_DIR}/
wget http://images.cocodataset.org/zips/train2017.zip -P ${DATA_DIR}/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ${DATA_DIR}/
unzip ${DATA_DIR}/val2017.zip -d ${DATA_DIR}/
unzip ${DATA_DIR}/train2017.zip -d ${DATA_DIR}/
unzip ${DATA_DIR}/annotations_trainval2017.zip -d ${DATA_DIR}/
python ../../../external/mmdetection/tools/coco_filter.py ${DATA_DIR}/annotations/instances_train2017.json ${DATA_DIR}/annotations/instances_train2017car.json --filter car --remap
python ../../../external/mmdetection/tools/coco_filter.py ${DATA_DIR}/annotations/instances_val2017.json ${DATA_DIR}/annotations/instances_val2017car.json --filter car --remap
sed -i "s/car/vehicle/g" ${DATA_DIR}/annotations/instances_val2017car.json
sed -i "s/car/vehicle/g" ${DATA_DIR}/annotations/instances_train2017car.json
```

### 3. Prepare annotation

Annotate dataset and save annotation to MSCOCO format with `vehicle` as the only one class.

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
