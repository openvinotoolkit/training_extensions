# Gesture Recognition

Models that are able to recognize gestures from live video stream on CPU.

* MS-ASL-100 gesture set (continuous scenario)

  | Model Name                        | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy  | Links                                                                                                                                                                                           | GPU_NUM |
  | --------------------------------- | ------------------- | --------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
  | s3d-rgb-mobilenet-v3-stream-msasl | 6.66	              | 4.133     | 84.7%           | [model template](s3d-rgb-mobilenet-v3-stream-msasl/template.yaml), [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-msasl1000.pth) | 2       |

* Jester-27 gesture set (continuous scenario)

  | Model Name                         | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy | Links                                                                                                                                                                                          | GPU_NUM |
  | ---------------------------------- | ------------------- | --------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
  | s3d-rgb-mobilenet-v3-stream-jester | 4.23	               | 4.133     | 93.58%         | [model template](s3d-rgb-mobilenet-v3-stream-jester/template.yaml), [snapshot](https://docs.google.com/uc?export=download&id=1lDm2qOxMRyXZW6y7owlQBv8SGvGIKpcX)                                | 4       |

## Datasets

Target datasets:
* [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads) - for MS-ASL-100 gesture models
* [Jester](https://20bn.com/datasets/jester) - for Jester-27 gesture models

## Training pipeline

### 1. Change a directory in your terminal to action_recognition.

```bash
cd models/action_recognition
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
export MODEL_TEMPLATE=`realpath ./model_templates/gesture-recognition/s3d-rgb-mobilenet-v3-stream-msasl/template.yaml`
export WORK_DIR=/tmp/my_model
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 3. Prepare data

Target datasets:
* To prepare MS-ASL data follow instructions: [DATA_MSASL.md](./DATA_MSASL.md).
* To prepare JESTER data follow instructions: [DATA_JESTER.md](./DATA_JESTER.md).

### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training and Fine-tuning

Try both following variants and select the best one:

* **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
* **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

If you would like to start **training** from pre-trained weights use `--load-weights` parameter with `imagenet1000-kinetics700-snapshot.pth` (you can download it [here](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-kinetics700.pth) for any s3d-rgb-mobilenet-v3-stream-XXX model).

If you would like to start **fine-tuning** from pre-trained weights use `--load-weights` parameter with `snapshot.pth`.

```bash
python train.py \
   --load-weights ${WORK_DIR}/snapshot.pth \
   --train-ann-files ${TRAIN_ANN_FILE} \
   --train-data-roots ${TRAIN_DATA_ROOT} \
   --val-ann-files ${VAL_ANN_FILE} \
   --val-data-roots ${VAL_DATA_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/outputs
```

> **NOTE**: It's recommended during fine-tuning to decrease the `--base-learning-rate` parameter compared with default value (see `${MODEL_TEMPLATE}`) to prevent from forgetting during the first iterations.

Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

### 6. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute mean accuracy metric run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 7. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar RGB format.

### 8. Demo

OpenVINO™ provides the Gesture Recognition demo, which is able to use the converted model. See details in the [demo](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/demos/gesture_recognition_demo/python).
