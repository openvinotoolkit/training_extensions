# Instance Segmentation

Models that are able to instantiate segmentation.

| Model Name |  Input resolution (HxW) | Complexity (GFLOPs) | Size (Mp) | Bbox AP @ [IoU=0.50:0.95] | Segm AP @ [IoU=0.50:0.95] | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| instance-segmentation-0904 |  384x416 | 41.36 | 29.7471 | 32.9 | 29.1 |  [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-0904-0912.pth), [model_template](./instance-segmentation-0904/template.yaml) | 2 |
| instance-segmentation-0912 | 512x544 | 66.49 | 29.7478 | 35.6 | 31.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-0904-0912.pth), [model_template](./instance-segmentation-0912/template.yaml) | 2 |
| instance-segmentation-0228 | 608x608 | 147.19 | 49.4374 | 39.0 | 33.9 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-0228.pth), [model_template](./instance-segmentation-0228/template.yaml) | 2 |
| instance-segmentation-0002 | 768x1024 | 423.02 | 47.58 | 40.8 | 36.9 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-0002.pth), [model_template](./instance-segmentation-0002/template.yaml) | 8 |
| instance-segmentation-0091 | 800x1344 | 828.45 | 100.1455 | 45.8 | 39.7 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-0091.pth), [model_template](./instance-segmentation-0091/template.yaml) | 8 |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 0. Change a directory in your terminal to instance_segmentation and activate venv.

```bash
cd models/instance_segmentation
```
If You have not created virtual environment yet:
```bash
./init_venv.sh
```
Activate virtual environment:
```bash
. venv/bin/activate
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/coco-instance-segmentation/instance-segmentation-0904/template.yaml`
export WORK_DIR=/tmp/my_model
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Collect dataset

Download the [COCO](https://cocodataset.org/#home) dataset and make the following
structure of the `../../data` directory:

```
data
├── coco
    ├── annotations
    ├── train2017
    ├── val2017
    ├── test2017
```

### 3. Prepare annotation

```bash
export INST_SEGM_DIR=`pwd`
export TRAIN_ANN_FILE="${INST_SEGM_DIR}/../../data/coco/annotations/instances_train2017.json"
export TRAIN_IMG_ROOT="${INST_SEGM_DIR}/../../data/coco/train2017"
export VAL_ANN_FILE="${INST_SEGM_DIR}/../../data/coco/annotations/instances_val2017.json"
export VAL_IMG_ROOT="${INST_SEGM_DIR}/../../data/coco/val2017"
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

   * If you would like to start **fine-tuning** from pre-trained weights use `--resume-from` parameter and value of `--epochs` have to exceed the value stored inside `${MODEL_TEMPLATE}` file, otherwise training will be ended immediately. Here we add `5` additional epochs.

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

You can also save images with predicted bounding boxes using `--save-output-to` parameter.

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --save-output-to ${WORK_DIR}/output_images
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

### 8. Validation of IR

Instead of passing `snapshot.pth` you need to pass path to `model.bin` (or `model.xml`).

```bash
python eval.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```
