# Quick Start Guide

## Prerequisites

Current version of OTX was tested under following environments

- Ubuntu 20.04
- Python 3.8.x
- (Opional) To use the NVidia GPU for the training: [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)

> **_Note:_** If using CUDA, make sure you are using a proper driver version. To do so, use `ls -la /usr/local | grep cuda`. If necessary, [install CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal) (requires 'sudo' permission) and select it with `export CUDA_HOME=/usr/local/cuda-11.1`.

## Setup OpenVINO™ Training Extensions

### Install OTX for users

1. Clone the training_extensions repository with the following commands:

```bash
git clone https://github.com/openvinotoolkit/training_extensions.git
cd training_extensions
git checkout develop
```

1. Set up a virtual environment

```bash
# Create virtual env.
python -m venv .venv

# Activate virtual env.
source .venv/bin/activate
```

1. Install prerequisite dependencies with:

Install pytorch according to your system environment.
Refer to the [official inatllation guide](https://pytorch.org/get-started/previous-versions/)

> **_Important note:_** Currently, only torch==1.3.1 was fully validated. torch==2.x will be supported soon. (Earlier versions are not supported due to security issues)

```bash
# Example install command for torch==1.13.1+cu116
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

(Optional) You may also want to use Jupyter notebooks or OTX CLI tools:

```bash
pip install notebook; cd ote_cli/notebooks/; jupyter notebook
```

> **_Important note:_** You should confirm that the Python version that installed on your machine should be 3.8.X. For the future release of OTX will support wide range of the Python version.

1. Then, install OTX package

```bash
# Install from local source in development mode
pip install -e .[full]

# Or, you can install from PyPI
pip install otx  # Will be available from otx==1.0.0
```

1. Once the package is installed to the virtual environment, you can use the
   `otx` command line interface to perform various commands for templates related to the chosen task type, described in [OTX CLI commands](#otx-cli-commands) on that virutal environment.

### Install OTX for developers

#### Run tests

```bash
# Install development prerequisite
python -m pip install requirements/dev.txt

# Run tests
tox -e pre-merge
tox -e pre-merge
```

#### Development environment

```bash
# Create development environment
# -- need to fix '38' in tox.ini if other python version needed
tox devenv -e pre-merge venv/otx
source venv/otx/bin/activate

# Fix code
# ... fixes will be directly applied to editable package

# Run test
pytest tests/unit
```

## OTX CLI commands

### Find

`find` lists model templates available for the given virtual environment.

```bash
(otx) ...$ otx find --help
usage: otx find [-h] [--root ROOT] [--task_type {classification,detection,segmentation,instance_segmantation,rotated_detection,anomaly_classification,anomaly_detection,anomaly_segmentation}]
                [--experimental]

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           A root dir where templates should be searched.
  --task_type {classification,detection,segmentation,instance_segmantation,rotated_detection,anomaly_classification,anomaly_detection,anomaly_segmentation}
  --experimental
```

```bash
# Example to find templates for the detection task
(otx) ...$ otx find --task_type detection
- id: Custom_Object_Detection_Gen3_SSD
  name: SSD
  path: otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml
  task_type: DETECTION
- id: Custom_Object_Detection_YOLOX
  name: YOLOX
  path: otx/algorithms/detection/configs/detection/cspdarknet_yolox/template.yaml
  task_type: DETECTION
- id: Custom_Object_Detection_Gen3_ATSS
  name: ATSS
  path: otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
  task_type: DETECTION
```

### Training

`train` trains a model (a particular model template) on a dataset and saves results in two files:

- `weights.pth` - a model snapshot
- `label_schema.json` - a label schema used in training, created from a dataset

These files can be used by other commands: `export`, `eval`, and `demo`.

`train` command requires `template` as a positional arguement. it could be taken from the output of the `find` command above.

```bash
otx train template
```

And with the `--help` command along with `template`, you can list additional information, such as its parameters common to all model templates and model-specific hyper parameters.

#### Common parameters

```bash
# Command example to get common paramters to any model templates
(otx) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx train [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS [--load-weights LOAD_WEIGHTS] --save-model-to SAVE_MODEL_TO
                 [--enable-hpo] [--hpo-time-ratio HPO_TIME_RATIO]
                 template {params} ...

positional arguments:
  template
  {params}              sub-command help
    params              Hyper parameters defined in template file.

optional arguments:
  -h, --help            show this help message and exit
  --train-ann-files TRAIN_ANN_FILES
                        Comma-separated paths to training annotation files.
  --train-data-roots TRAIN_DATA_ROOTS
                        Comma-separated paths to training data folders.
  --val-ann-files VAL_ANN_FILES
                        Comma-separated paths to validation annotation files.
  --val-data-roots VAL_DATA_ROOTS
                        Comma-separated paths to validation data folders.
  --load-weights LOAD_WEIGHTS
                        Load only weights from previously saved checkpoint
  --resume-from RESUME_FROM
                         Resume training from previously saved checkpoint
  --save-model-to SAVE_MODEL_TO
                        Location where trained model will be stored.
  --enable-hpo          Execute hyper parameters optimization (HPO) before training.
  --hpo-time-ratio HPO_TIME_RATIO
                        Expected ratio of total time to run HPO to time taken for full fine-tuning.
```

#### Model template-specific parameters

command example:

```bash
# command example to get tamplate-specific parameters
(otx) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml params --help
usage: otx train template params [-h] [--learning_parameters.batch_size BATCH_SIZE] [--learning_parameters.learning_rate LEARNING_RATE] [--learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS]
                                 [--learning_parameters.num_iters NUM_ITERS] [--learning_parameters.enable_early_stopping ENABLE_EARLY_STOPPING] [--learning_parameters.early_stop_start EARLY_STOP_START]
                                 [--learning_parameters.early_stop_patience EARLY_STOP_PATIENCE] [--learning_parameters.early_stop_iteration_patience EARLY_STOP_ITERATION_PATIENCE]
                                 [--learning_parameters.use_adaptive_interval USE_ADAPTIVE_INTERVAL] [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD]
                                 [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD] [--nncf_optimization.enable_quantization ENABLE_QUANTIZATION]
                                 [--nncf_optimization.enable_pruning ENABLE_PRUNING] [--nncf_optimization.pruning_supported PRUNING_SUPPORTED] [--tiling_parameters.enable_tiling ENABLE_TILING]
                                 [--tiling_parameters.enable_adaptive_params ENABLE_ADAPTIVE_PARAMS] [--tiling_parameters.tile_size TILE_SIZE] [--tiling_parameters.tile_overlap TILE_OVERLAP]
                                 [--tiling_parameters.tile_max_number TILE_MAX_NUMBER]

optional arguments:
  -h, --help            show this help message and exit
  --learning_parameters.batch_size BATCH_SIZE
                        header: Batch size
                        type: INTEGER
                        default_value: 8
                        max_value: 512
                        min_value: 1
  --learning_parameters.learning_rate LEARNING_RATE
                        header: Learning rate
                        type: FLOAT
                        default_value: 0.01
                        max_value: 0.1
                        min_value: 1e-07
  --learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS
                        header: Number of iterations for learning rate warmup
                        type: INTEGER
                        default_value: 3
                        max_value: 10000
                        min_value: 0
...
```

#### Command example of the training

```bash
(otx) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --train-ann-file data/airport/annotation_person_train.json  --train-data-roots data/airport/train/ --val-ann-files data/airport/annotation_person_val.json --val-data-roots data/airport/val/ --save-model-to outputs
...

---------------iou_thr: 0.5---------------

+--------+-----+------+--------+-------+
| class  | gts | dets | recall | ap    |
+--------+-----+------+--------+-------+
| person | 0   | 2000 | 0.000  | 0.000 |
+--------+-----+------+--------+-------+
| mAP    |     |      |        | 0.000 |
+--------+-----+------+--------+-------+
2022-11-17 11:08:15,245 | INFO : run task done.
2022-11-17 11:08:15,318 | INFO : Inference completed
2022-11-17 11:08:15,319 | INFO : called evaluate()
2022-11-17 11:08:15,334 | INFO : F-measure after evaluation: 0.8809523809523808
2022-11-17 11:08:15,334 | INFO : Evaluation completed
Performance(score: 0.8809523809523808, dashboard: (1 metric groups))
```

### Exporting

`export` exports a trained model to the OpenVINO format in order to efficiently run it on Intel hardware.

With the `--help` command, you can list additional information, such as its parameters common to all model templates:
command example:

```bash
(otx) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx export [-h] --load-weights LOAD_WEIGHTS --save-model-to SAVE_MODEL_TO template

positional arguments:
  template

optional arguments:
  -h, --help            show this help message and exit
  --load-weights LOAD_WEIGHTS
                        Load weights from saved checkpoint for exporting
  --save-model-to SAVE_MODEL_TO
                        Location where exported model will be stored.
```

#### Command example of the exporting

The command below performs exporting to the [trained model](#command-example-of-the-training) `outputs/weights.pth` in previous section and save exported model to the `outputs/ov/` folder.

```bash
(otx) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --load-weights outputs/weights.pth --save-model-to outputs/ov
...
[ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
Find more information about API v2.0 and IR v11 at https://docs.openvino.ai
2022-11-21 15:40:06,534 | INFO : Exporting completed
2022-11-21 15:40:06,534 | INFO : run task done.
2022-11-21 15:40:06,538 | INFO : Exporting completed
```

### Optimization

`optimize` optimizes a model using NNCF or POT depending on the model format.

- NNCF optimization used for trained snapshots in a framework-specific format such as checkpoint (pth) file from Pytorch
- POT optimization used for models exported in the OpenVINO IR format

With the `--help` command, you can list additional information.
command example:

```bash
(otx) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx optimize [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS --load-weights LOAD_WEIGHTS --save-model-to SAVE_MODEL_TO
                    [--save-performance SAVE_PERFORMANCE]
                    template {params} ...

positional arguments:
  template
  {params}              sub-command help
    params              Hyper parameters defined in template file.

optional arguments:
  -h, --help            show this help message and exit
  --train-ann-files TRAIN_ANN_FILES
                        Comma-separated paths to training annotation files.
  --train-data-roots TRAIN_DATA_ROOTS
                        Comma-separated paths to training data folders.
  --val-ann-files VAL_ANN_FILES
                        Comma-separated paths to validation annotation files.
  --val-data-roots VAL_DATA_ROOTS
                        Comma-separated paths to validation data folders.
  --load-weights LOAD_WEIGHTS
                        Load weights of trained model (for NNCF) or exported OpenVINO model (for POT)
  --save-model-to SAVE_MODEL_TO
                        Location where trained model will be stored.
  --save-performance SAVE_PERFORMANCE
                        Path to a json file where computed performance will be stored.
```

#### Command example for optimizing a PyTorch model (.pth) with OpenVINO NNCF

The command below performs optimization to the [trained model](#command-example-of-the-training) `outputs/weights.pth` in previous section and save optimized model to the `outputs/nncf` folder.

```bash
(otx) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --train-ann-files data/airport/annotation_person_train.json --train-data-roots data/airport/train/ --val-ann-files data/airport/annotation_person_val.json --val-data-roots data/airport/val/ --load-weights outputs/weights.pth --save-model-to outputs/nncf --save-performance outputs/nncf/performance.json
```

#### Command example for optimizing OpenVINO model (.xml) with OpenVINO POT

The command below performs optimization to the [exported model](#command-example-of-the-exporting) `outputs/ov/openvino.xml` in previous section and save optimized model to the `outputs/ov/pot` folder.

```bash
(otx) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --train-ann-files data/airport/annotation_person_train.json --train-data-roots data/airport/train/ --val-ann-files data/airport/annotation_person_val.json --val-data-roots data/airport/val/ --load-weights outputs/ov/openvino.xml --save-model-to outputs/ov/pot --save-performance outputs/ov/pot/performance.json
```

### Evaluation

`eval` runs evaluation of a model on the particular dataset.

With the `--help` command, you can list additional information, such as its parameters common to all model templates:
command example:

```bash
(otx) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx eval [-h] --test-ann-files TEST_ANN_FILES --test-data-roots TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS [--save-performance SAVE_PERFORMANCE] template {params} ...

positional arguments:
  template
  {params}              sub-command help
    params              Hyper parameters defined in template file.

optional arguments:
  -h, --help            show this help message and exit
  --test-ann-files TEST_ANN_FILES
                        Comma-separated paths to test annotation files.
  --test-data-roots TEST_DATA_ROOTS
                        Comma-separated paths to test data folders.
  --load-weights LOAD_WEIGHTS
                        Load weights to run the evaluation. It could be a trained/optimized model or exported model.
  --save-performance SAVE_PERFORMANCE
                        Path to a json file where computed performance will be stored.
```

> **_Note:_** Work-In-Progress for `params` argument.

#### Command example of the evaluation

The command below performs evaluation to the [trained model](#command-example-of-the-training) `outputs/weights.pth` in previous section and save result performance to the `outputs/performance.json` file.

```bash
(otx) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --test-ann-files data/airport/annotation_person_val.json --test-data-roots data/airport/val/ --load-weights outputs/weights.pth --save-performance outputs/performance.json
...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 10/10, 7.9 task/s, elapsed: 1s, ETA:     0s
---------------iou_thr: 0.5---------------

+--------+-----+------+--------+-------+
| class  | gts | dets | recall | ap    |
+--------+-----+------+--------+-------+
| person | 0   | 2000 | 0.000  | 0.000 |
+--------+-----+------+--------+-------+
| mAP    |     |      |        | 0.000 |
+--------+-----+------+--------+-------+
2022-11-21 15:30:04,695 | INFO : run task done.
2022-11-21 15:30:04,734 | INFO : Inference completed
2022-11-21 15:30:04,734 | INFO : called evaluate()
2022-11-21 15:30:04,746 | INFO : F-measure after evaluation: 0.8799999999999999
2022-11-21 15:30:04,746 | INFO : Evaluation completed
Performance(score: 0.8799999999999999, dashboard: (1 metric groups))
```

### Demonstrate

`demo` runs model inference on images, videos, or webcam streams in order to see how it works with user's data

> **_Note:_** `demo` command requires GUI backend to your system for displaying inference results.
>
> **_Note:_** The model optimzied with `NNCF` is not supported for the `demo` command.

With the `--help` command, you can list additional information, such as its parameters common to all model templates:
command example:

```bash
(otx) ...$ otx demo otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx demo [-h] -i INPUT --load-weights LOAD_WEIGHTS [--fit-to-size FIT_TO_SIZE FIT_TO_SIZE] [--loop] [--delay DELAY] [--display-perf] template {params} ...

positional arguments:
  template
  {params}              sub-command help
    params              Hyper parameters defined in template file.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Source of input data: images folder, image, webcam and video.
  --load-weights LOAD_WEIGHTS
                        Load weights to run the evaluation. It could be a trained/optimized model (POT only) or exported model.
  --fit-to-size FIT_TO_SIZE FIT_TO_SIZE
                        Width and Height space-separated values. Fits displayed images to window with specified Width and Height. This options applies to result visualisation only.
  --loop                Enable reading the input in a loop.
  --delay DELAY         Frame visualization time in ms.
  --display-perf        This option enables writing performance metrics on displayed frame. These metrics take into account not only model inference time, but also frame reading, pre-processing and post-processing.
```

#### Command example of the demostration

The command below performs demonstration to the [optimized model](#command-example-for-optimizing-openvino-model-xml-with-openvino-pot) `outputs/ov/pot/openvino.xml` in previous section with images in the given input folder.

```bash
(otx) ...$ otx demo otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --input data/airport/val/ --load-weights outputs/ov/pot/openvino.xml --display-perf --delay 1000
...
[ INFO ] OpenVINO inference completed
```

> **_Note:_** The inference results with a model will be display to the GUI window with 1 second interval. If you execute this command from the remote environment (e.g., using text-only SSH via terminal) without having remote GUI client software, you can meet some error message from this command.

### Deployment

`deploy` creates openvino.zip with a self-contained python package, a demo application, and an exported model.

With the `--help` command, you can list additional information, such as its parameters common to all model templates:
command example:

```bash
(otx) ...$ otx deploy otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml --help
usage: otx deploy [-h] --load-weights LOAD_WEIGHTS [--save-model-to SAVE_MODEL_TO] template

positional arguments:
  template

optional arguments:
  -h, --help            show this help message and exit
  --load-weights LOAD_WEIGHTS
                        Load model's weights from.
  --save-model-to SAVE_MODEL_TO
                        Location where openvino.zip will be stored.
```

---

\* Other names and brands may be claimed as the property of others.
