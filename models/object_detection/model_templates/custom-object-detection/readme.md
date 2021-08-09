# Custom object detector

Custom object detectors are lightweight object detection models that have been pre-trained on MS COCO object detection dataset.
It is assumed that one will use these pre-trained models as starting points in order to train specific object detection models (e.g. 'cat' and 'dog' detection).
*NOTE* There was no goal to train top-scoring lightweight 80 class (MS COCO classes) detectors here,
but rather provide pre-trained checkpoints for further fine-tuning on a target dataset.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v2-2s_ssd-256x256 | 0.86 | 1.99 | 11.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-256x256.pth), [model template](./mobilenet_v2-2s_ssd-256x256/template.yaml) | 3 |
| mobilenet_v2-2s_ssd-384x384 | 1.92 | 1.99 | 13.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-384x384.pth), [model template](./mobilenet_v2-2s_ssd-384x384/template.yaml) | 3 |
| mobilenet_v2-2s_ssd-512x512 | 3.42 | 1.99 | 12.7 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-512x512.pth), [model template](./mobilenet_v2-2s_ssd-512x512/template.yaml) | 3 |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 1. Change a directory in your terminal to object_detection.

```bash
cd models/object_detection
```
If you have not created virtual environment yet:
```bash
./init_venv.sh
```
Activate virtual environment:
```bash
source venv/bin/activate
```

### 2. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml`
export WORK_DIR=/tmp/my_model
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 3. Collect dataset

You can train a model on existing toy dataset `training_extensions/data/airport`. Obviously such dataset is not sufficient for training good enough model.

### 4. Prepare annotation

The existing toy dataset has annotation in the Common Objects in Context (COCO) format so it is enough to get started.

```bash
export OBJ_DET_DIR=`pwd`
export TRAIN_ANN_FILE="${OBJ_DET_DIR}/../../data/airport/annotation_example_train.json"
export TRAIN_IMG_ROOT="${OBJ_DET_DIR}/../../data/airport/train"
export VAL_ANN_FILE="${OBJ_DET_DIR}/../../data/airport/annotation_example_val.json"
export VAL_IMG_ROOT="${OBJ_DET_DIR}/../../data/airport/val"
```

### 5. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 6. Training

Since custom detection model templates rather than ready-to-use models (though technically one can use them as they are) are provided it is needed to define `classes`.

```bash
export CLASSES="vehicle,person,non-vehicle"
```

### 7. Training

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

### 8. Evaluation

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

### 9. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

For SSD networks an alternative OpenVINO™ representation is saved automatically to `${WORK_DIR}/export/alt_ssd_export` folder.
SSD model exported in such way will produce a bit different results (non-significant in most cases),
but it also might be faster than the default one. As a rule SSD models in [Open Model Zoo](https://github.com/opencv/open_model_zoo/) are exported using this option.

### 10. Validation of IR

Instead of passing `snapshot.pth` you need to pass path to `model.bin`.

```bash
python eval.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml \
   --classes ${CLASSES}
```

### 11. Optimization

The trained models can be optimized -- compressed by [NNCF](https://github.com/openvinotoolkit/nncf) framework.

To use NNCF to compress a custom object detection model, you should go to the root folder of this git repository
and install compression requirements in your virtual environment by the command
```bash
pip install -r external/mmdetection/requirements/nncf_compression.txt
```

At the moment, only one compression method is supported for custom object detection models:
[int8 quantization](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

To compress the model, 'compress.py' script should be used.

Please, note that NNCF framework requires a dataset for compression, since it makes several steps of fine-tuning after
compression to restore the quality of the model, so the command line parameters of the script `compress.py` are closer
to the command line parameter of the training script (see the section "7. Training" stated above):
```
      python compress.py \
         --load-weights ${SNAPSHOT} \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_IMG_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_IMG_ROOT} \
         --save-checkpoints-to outputs \
         --nncf-quantization
```
Note that the number of epochs required for NNCF compression should not be set by command line parameter, since it is
calculated by the script `compress.py` itself.

The compressed model can be evaluated and exported to the OpenVINO™ format by the same commands as non-compressed model,
see the sections 8 and 9 above.
