# Custom Action Recognition

Models that are trained on the Kinetics-700 and YouTube-8M-Segments datasets simultaneously and able to recognize actions from live video stream on CPU.

Performance results table:

| Model Name                  | Complexity (GFLOPs) | Size (Mp) | UCF-101 Top-1 accuracy | ActivityNet v1.3 Top-1 Accuracy | Jester-27 Top-1 accuracy | MS-ASL-1000 Top-1 accuracy | Links                                                                                                                                                                                                                                                              |
| --------------------------- | ------------------- | --------- | ---------------------- | ------------------------------- | ------------------------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| s3d-rgb-mobilenet-v3        | 6.65                | 4.116     | 93.79%                 | 64.09%                          | 93.79%                   | 41.20%                     | [model template](s3d-rgb-mobilenet-v3/template.yaml), [kinetics-700 snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_action_recognition/s3d-mobilenetv3-large-kinetics700.pth)                                |
| x3d-rgb-mobilenet-v3-lgd-gc | 4.74                | 4.472     | 93.63%                 | 64.19%                          | 95.36%                   | 20.60%                     | [model template](x3d-rgb-mobilenet-v3-lgd-gc/template.yaml), [kinetics-700 snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_action_recognition/x3d-mobilenetv3-large-kinetics700-youtube8msegments-fixed.pth) |

> **NOTE**: The top-1 accuracy metric is calculated as a single clip/crop per video to demonstrate the real model performance on the reported model complexity.

## Datasets

The following datasets were used in experiments:
* [UCF-101](https://export.arxiv.org/abs/1212.0402)
* [ActivityNet v1.3](http://activity-net.org/download.html)
* [MS-ASL](https://www.microsoft.com/en-us/research/project/ms-asl/#!downloads)
* [Jester](https://20bn.com/datasets/jester)
* [Kinetics-700](https://deepmind.com/research/open-source/kinetics)
* [YouTube-8M-Segments](https://research.google.com/youtube8m)

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
export MODEL_TEMPLATE=`realpath ./model_templates/custom-action-recognition/s3d-rgb-mobilenet-v3/template.yaml`
export WORK_DIR=/tmp/my_model
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 3. Prepare data

The training script assumes the data for the action recognition is provided as raw frames stored in directories where each directory represents a single video source.
Additionally we assume that the data is split on train/val subsets.
The annotation file consists of lines where each line represents single video source in the following format:
```
<rel_path_to_video_dir> <label_id> <start_video_frame_id> <end_video_frame_id> <start_clip_frame_id> <end_clip_frame_id> <video_fps>
```

where:
* `<rel_path_to_video_dir>` - relative path to the directory with dumped video frames.
* `<label_id>` - ID of ground-truth class.
* `<start_video_frame_id>`/`<end_video_frame_id>` - start/end frame IDs of the whole video.
* `<start_clip_frame_id>`/`<end_clip_frame_id>` - start/end frame IDS of the action inside the video.
* `<video_fps>` - the video frame-rate (frames-per-second).

> **NOTE**: Training/validation scripts expects action class IDs instead of class labels. So, action labels should be manually sorted and converted in the appropriate class IDs.

If you have the data in the video format (videos instead of dumped frames) you may use the following script to dump frames and generate the annotation file in the proper format.
It assumes you have videos in `${DATA_DIR}/videos` directory and the appropriate annotation file with video names and class IDs.
To dump frames and convert annotation run the following script:

```bash
python ./tools/dump_frames.py \
   --videos_dir ${DATA_DIR}/videos \
   --annotation ${DATA_DIR}/train.txt ${DATA_DIR}/val.txt \
   --output_dir ${DATA_DIR}
```

where `${DATA_DIR}/train.txt` and `${DATA_DIR}/val.txt` - annotation files where each line represents single video source in the following format:

```
<rel_path_to_video_file> <label_id>
```

Additionally to enable the CrossNorm augmentation please download the [file](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_action_recognition/mean_std_list.txt) with dumped mean and variance pairs and place it into the `${DATA_DIR}`.

Finally, the `${DATA_DIR}` directory should be like this:

```
${DATA_DIR}
├── custom_dataset
│   ├── rawframes
│   │   ├── video_name_0
│   |   |   ├── 00001.jpg
│   |   |   └── ...
│   |   └── ...
│   ├── val.txt
│   └── train.txt
└── mean_std_list.txt
```

After the data was arranged, export the variables required for launching training and evaluation scripts:

```bash
export DATA_DIR=${WORK_DIR}/data
export TRAIN_ANN_FILE=train.txt
export TRAIN_DATA_ROOT=${DATA_DIR}
export VAL_ANN_FILE=val.txt
export VAL_DATA_ROOT=${DATA_DIR}
export TEST_ANN_FILE=val.txt
export TEST_DATA_ROOT=${DATA_DIR}
```

### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training and Fine-tuning

Try both following variants and select the best one:

* **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
* **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

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
