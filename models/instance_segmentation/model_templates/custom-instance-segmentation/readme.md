# Custom Instance Segmentation

Custom instance segmentation models are lightweight models that have been pre-trained on MS COCO instance segmentation dataset.
It is assumed that one will use these pre-trained models as starting points in order to train specific instance segmentation models (e.g. 'cat' and 'dog' detection).
*NOTE* There was no goal to train top-scoring lightweight 80 class (MS COCO classes) detectors here,
but rather provide pre-trained checkpoints and a good training config for further fine-tuning on a target dataset.

| Model Name | Complexity (GFLOPs) | Size (Mp) | Bbox AP @ [IoU=0.50:0.95] | Segm AP @ [IoU=0.50:0.95] | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- | --- |
| efficientnet_b2b-mask_rcnn-480x480 | 14.8 | 10.27 | 34.1 | 29.4 | [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-480x480.pth), [model template](efficientnet_b2b-mask_rcnn-480x480/template.yaml) | 1 |
| efficientnet_b2b-mask_rcnn-576x576 | 26.92 | 13.27 | 35.2 | 31.0 | [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/instance_segmentation/v2/efficientnet_b2b-mask_rcnn-576x576.pth), [model template](efficientnet_b2b-mask_rcnn-576x576/template.yaml) | 1 |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 0. Change a directory in your terminal to instance_segmentation and activate venv.

```bash
cd <training_extensions>/pytorch_toolkit/instance_segmentation
```
If You have not created virtual environment yet:
```bash
./init_venv.sh
```
Else:
```bash
. venv/bin/activate
```
or if You use conda:
```bash
conda activate <environment_name>
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/coco-instance-segmentation/instance-segmentation-0904/template.yaml`
export WORK_DIR=/tmp/my_model
python ../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
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

### 4. Change a current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training

Since custom instance segmentation model templates rather than ready-to-use models (though technically one can use them as they are) are provided it is needed to define `classes`.

```bash
export CLASSES="person,car"
```

### 6. Training

To start training from pre-trained weights use `--load-weights` pararmeter.

```bash
python train.py \
   --load-weights ${WORK_DIR}/snapshot.pth \
   --train-ann-files ${TRAIN_ANN_FILE} \
   --train-data-roots ${TRAIN_IMG_ROOT} \
   --val-ann-files ${VAL_ANN_FILE} \
   --val-data-roots ${VAL_IMG_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/outputs \
   --classes ${CLASSES}
```

Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

### 7. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute MS-COCO metrics and save computed values to `${WORK_DIR}/metrics.yaml` run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --classes ${CLASSES}
```

You can also save images with predicted bounding boxes using `--save-output-to` parameter.

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --save-output-to ${WORK_DIR}/output_images \
   --classes ${CLASSES}
```

### 8. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

### 9. Validation of IR

Instead of passing `snapshot.pth` you need to pass path to `model.bin` (or `model.xml`).

```bash
python eval.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --classes ${CLASSES}
```
