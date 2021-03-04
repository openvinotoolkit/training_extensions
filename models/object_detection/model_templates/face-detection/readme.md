# Face Detection

Models that are able to detect faces.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | AP for faces > 64x64 (%) | WiderFace Easy (%) | WiderFace Medium (%) | WiderFace Hard (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face-detection-0200 | 0.82 | 1.83 | 16.0 | 86.743 | 82.917 | 76.198 | 41.443 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth), [model template](./face-detection-0200/template.yaml) | 2 |
| face-detection-0202 | 1.84 | 1.83 | 20.3 | 91.938 | 89.382 | 83.919 | 50.189 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0202.pth), [model template](./face-detection-0202/template.yaml) | 2 |
| face-detection-0204 | 2.52 | 1.83 | 21.4 | 92.888 | 90.453 | 85.448 | 52.091 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0204.pth), [model template](./face-detection-0204/template.yaml) | 4 |
| face-detection-0205 | 2.94 | 2.02 | 21.6 | 93.566 | 92.032 | 86.717 | 54.055 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0205.pth), [model template](./face-detection-0205/template.yaml) | 4 |
| face-detection-0206 | 340.06 | 63.79 | 34.2 | 94.274 | 94.281 | 93.207 | 84.439 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0206.pth), [model template](./face-detection-0206/template.yaml) | 8 |
| face-detection-0207 | 1.04 | 0.81 | 17.2 | 88.17 | 84.406 | 76.748 | 43.452 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0207.pth), [model template](./face-detection-0207/template.yaml) | 1 |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 1. Change a directory in your terminal to object_detection.

```bash
cd models/object_detection
```
If You have not created virtual environment yet:
```bash
./init_venv.sh
```
Activate virtual environment:
```bash
source venv/bin/activate
```

### 2. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/face-detection/face-detection-0200/template.yaml`
export WORK_DIR=/tmp/my_model
export SNAPSHOT=${WORK_DIR}/snapshot.pth
export ADD_EPOCHS=1
export EPOCHS_NUM=$((`cat ${MODEL_TEMPLATE} | grep epochs | tr -dc '0-9'` + ${ADD_EPOCHS}))
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 3. Collect dataset

In this particular toy example we would like to demonstrate an ability of training code to improve quality of a model on particular dataset during fine-tuning. That's why training and validation datasets are be represented by the same set of images. Training images are stored in ${TRAIN_IMG_ROOT} together with ${TRAIN_ANN_FILE} annotation file. The annotation file has been created manually using [CVAT](https://github.com/openvinotoolkit/cvat).

If you would like to work with bigger datasets please refer to this [section](datasets.md), if not:

```bash
export TRAIN_ANN_FILE=`pwd`/../../data/airport/annotation_faces_train.json
export TRAIN_IMG_ROOT=`pwd`/../../data/airport/
export VAL_ANN_FILE=${TRAIN_ANN_FILE}
export VAL_IMG_ROOT=${TRAIN_IMG_ROOT}
```

### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```
### 5. Export pretrained PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${SNAPSHOT} \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

### 6. Run demo with exported model.

You need to pass a path to `model.bin` and index of your web cam.

```bash
python visualize.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --video 0
```

### 7. Evaluation of exported model.

Instead of passing `snapshot.pth` you need to pass path to `model.bin`.

```bash
python eval.py \
   --load-weights ${WORK_DIR}/export/model.bin \
   --test-ann-files ${VAL_ANN_FILE} \
   --test-data-roots ${VAL_IMG_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 8. Training and Fine-tuning

Try both following variants and select the best one:

   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.
   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.


   * If you would like to start **fine-tuning** from pre-trained weights use `--resume-from` parameter and value of `--epochs` have to exceed the value stored inside `${MODEL_TEMPLATE}` file, otherwise training will be ended immediately. Here we add `2` additional epochs.

      ```bash
      python train.py \
         --resume-from ${SNAPSHOT} \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_IMG_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_IMG_ROOT} \
         --save-checkpoints-to ${WORK_DIR}/outputs \
         --epochs ${EPOCHS_NUM} \
      && export SNAPSHOT=${WORK_DIR}/outputs/latest.pth \
      && export EPOCHS_NUM=$((${EPOCHS_NUM} + ${ADD_EPOCHS}))
      ```

   * If you would like to start **training** from pre-trained weights use `--load-weights` pararmeter instead of `--resume-from`. Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

### 9. Compression

One can apply compression algorithms that are intented to optimize inference even more.
This can be done by runnning `compress.py` script with `--nncf-quantization` or `--nncf-sparsity` or both:

```bash
python compress.py \
   --load-weights ${SNAPSHOT} \
   --train-ann-files ${TRAIN_ANN_FILE} \
   --train-data-roots ${TRAIN_IMG_ROOT} \
   --val-ann-files ${VAL_ANN_FILE} \
   --val-data-roots ${VAL_IMG_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/compressed \
   --nncf-quantization
```
