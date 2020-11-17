# ASL Recognition

Models that are able to recognize ASL gestures (MS-ASL-100 gesture set) from live video stream on CPU.

| Model Name                  | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy (MS-ASL-100) | Links                                                                                                                                                                                       | GPU_NUM |
| --------------------------- | ------------------- | --------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| asl-recognition-0004        | 6.66	            | 4.133     | 84.7%                       | [model template](s3d-rgb-mobilenet-v3-stream-msasl/template.yaml), [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-msasl1000.pth) | 2       |

## Datasets

Target datasets:
* [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads)

> **Note**: Due to significant noise in the original annotation of MS-ASl dataset we use the cleaned version which includes:
>
> * Filtering invalid videos
> * Filtering invalid temporal crops
> * Enhancing temporal limits of gestures
> * Hiding text captions of presented gesture

Pre-training datasets (to get the best performance):
* [ImageNet-1000](http://image-net.org/download) (2D backbone pre-training)
* [Kinetics-700](https://deepmind.com/research/open-source/kinetics) (full model pre-training)

> **Note**: To skip the pre-training stage we provide the [S3D MobileNet-V3](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-kinetics700.pth) pre-trained on both ImageNet-1000 and Kinetics-700 datasets.

## Training pipeline

### 0. Change a directory in your terminal to action_recognition_2.

```bash
cd <training_extensions>/pytorch_toolkit/action_recognition_2
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/gesture_recognition/s3d-rgb-mobilenet-v3-stream-msasl/template.yaml`
export WORK_DIR=/tmp/my_model
python ../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 2. Download annotation

Download the [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads) annotation and unpack it to `${DATA_DIR}/msasl_data` folder.

```bash
export DATA_DIR=${WORK_DIR}/data
```

### 3. Download videos

Download MS-ASL videos using the unpacked annotation files (`MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`):

```bash
python3 ./model_templates/asl_recognition/tools/data/download_msasl_videos.py \
  -s ${DATA_DIR}/msasl_data/MSASL_train.json ${DATA_DIR}/msasl_data/MSASL_val.json ${DATA_DIR}/msasl_data/MSASL_test.json \
  -o ${DATA_DIR}/msasl_data/videos
```

### 4. Convert dataset

Extract frames and prepare annotation files by running the following command:

```bash
python3 ./model_templates/asl_recognition/tools/data/extract_msasl_frames.py \
  -s ${DATA_DIR}/msasl_data/MSASL_train.json ${DATA_DIR}/msasl_data/MSASL_val.json ${DATA_DIR}/msasl_data/MSASL_test.json \
  -v ${DATA_DIR}/msasl_data/videos \
  -o ${DATA_DIR}/msasl
```

Split annotation files by running the following commands:

```bash
python3 ./model_templates/asl_recognition/tools/data/split_msasl_annotation.py \
  -a ${DATA_DIR}/msasl/train.txt ${DATA_DIR}/msasl/val.txt ${DATA_DIR}/msasl/test.txt \
  -k 100
export TRAIN_ANN_FILE=train.txt
export TRAIN_DATA_ROOT=${DATA_DIR}
export VAL_ANN_FILE=val.txt
export VAL_DATA_ROOT=${DATA_DIR}
export TEST_ANN_FILE=test.txt
export TEST_DATA_ROOT=${DATA_DIR}
```

To get the most robust model it's recommended to enable the [mixup](https://arxiv.org/abs/1710.09412) augmentation by specifying the paths to images in `imagenet_train_list.txt` file.
Additionally you should enable MixUp by uncommenting appropriate line in `model.py` config.

In this repo we use ImageNet dataset but it's possible to use similar dataset with images. In case of other dataset you only need to create the `imagenet_train_list.txt` file with paths to images.
If you have downloaded ImageNet dataset place it in `${DATA_DIR}/imagenet` folder and dump image paths by running command:

```bash
python3 ./model_templates/asl_recognition/tools/data/get_imagenet_paths.py \
  ${DATA_DIR}/train \
  ${DATA_DIR}/imagenet_train_list.txt
```

Finally, the `${DATA_DIR}` directory should be like this:

   ```
   ${DATA_DIR}
   ├── msasl
   |   ├── global_crops
   |   │   ├── video_name_0
   |   │   |   ├── clip_0000
   |   |   |   |   ├── img_00001.jpg
   |   |   |   |   └── ...
   |   │   |   └── ...
   |   |   └── ...
   |   ├── val100.txt
   |   ├── test100.txt
   |   └── train1000.txt
   ├── imagenet
   |   └── train
   └── imagenet_train_list.txt
   ```

### 5. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 6. Training and Fine-tuning

Try both following variants and select the best one:

* **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
* **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

If you would like to start **training** from pre-trained weights use `--load-weights` parameter with `imagenet1000-kinetics700-snapshot.pth`.

```bash
python train.py \
   --load-weights ${WORK_DIR}/imagenet1000-kinetics700-snapshot.pth \
   --train-ann-files ${TRAIN_ANN_FILE} \
   --train-data-roots ${TRAIN_DATA_ROOT} \
   --val-ann-files ${VAL_ANN_FILE} \
   --val-data-roots ${VAL_DATA_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/outputs
```

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

### 7. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute mean accuracy metric run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 8. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar RGB format.

### 9. Demo

OpenVINO™ provides the Gesture Recognition demo, which is able to use the converted model. See details in the [demo](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/demos/python_demos/gesture_recognition_demo).
