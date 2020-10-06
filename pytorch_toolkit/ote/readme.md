# Model Templates

Model Templates defines training procedure and its interface for a given neural network topology.

## Directory structure

A single Model Template should consist of a Python script related to model training as well
as YAML file that will define template interface. Each model template is related to 4 scripts: `train.py`, `eval.py`, `export.py` and `quantize.py`. Model Templates may be placed in nested directories in the whole repository and will be detected
automatically by the Platform. Directories does not need to conform to any convention as in the example below:

```bash
/example_model_template
├── network
│   ├── __init__.py
│   └── utils.py
├── requirements.txt
├── template.yaml
├── eval.py
└── train.py
```

## template.yml

Each Model Template should be described by YAML file, which defines configuration supported by its
implementation. It may define which hyper-parameters are available, what output formats it can
generate, set of supported optimisations and so on.

Here is an example of `template.yaml`:

```bash
name: face-detection-0200
domain: Object Detection
problem: Face Detection
framework: OTEDetection v2.1.0.1
summary: Face Detection based on MobileNetV2 (SSD).
annotation_format: COCO
dependencies:
- sha256: 122e194b95ef631f578fac0dc88ba2be7b2ef9d06263ff6e7a69c846990a85f7
  size: 14904296
  source: https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth
  destination: snapshot.pth
- source: ../tools/train.py
  destination: train.py
- source: ../tools/eval.py
  destination: eval.py
- source: ../../../tools/export.py
  destination: export.py
- source: ../../../tools/quantize.py
  destination: quantize.py
- source: ../../../../../pytorch_toolkit/ote
  destination: ote
- source: ../../../../object_detection/oteod
  destination: oteod
- source: ../../requirements.txt
  destination: requirements.txt
max_nodes: 1
training_target:
- CPU
- GPU
inference_target:
- CPU
- iGPU
- VPU
hyper_parameters:
  basic:
    batch_size: 65
    base_learning_rate: 0.05
    epochs: 70
output_format:
  onnx:
    default: true
  openvino:
    default: true
    input_format: BGR
quantization: TBD
metrics:
- display_name: AP @ [IoU=0.50:0.95]
  key: ap
  unit: '%'
  value: 16.0
- display_name: AP for faces > 64x64
  key: ap_64x64
  unit: '%'
  value: 86.743
- display_name: WiderFace Easy
  key: widerface_e
  unit: '%'
  value: 82.917
- display_name: WiderFace Medium
  key: widerface_m
  unit: '%'
  value: 76.198
- display_name: WiderFace Hard
  key: widerface_h
  unit: '%'
  value: 41.443
- display_name: Size
  key: size
  unit: Mp
  value: 1.83
- display_name: Complexity
  key: complexity
  unit: GFLOPs
  value: 0.82
gpu_num: 2
config: model.py
estimated_batch_time: 1.0

```

## train.py
```bash
python ../tools/train.py -h
usage: train.py [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots
                TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES
                --val-data-roots VAL_DATA_ROOTS [--resume-from RESUME_FROM]
                [--load-weights LOAD_WEIGHTS]
                [--save-checkpoints-to SAVE_CHECKPOINTS_TO] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE]
                [--base-learning-rate BASE_LEARNING_RATE] [--gpu-num GPU_NUM]

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
  --resume-from RESUME_FROM
                        Resume training from previously saved checkpoint
  --load-weights LOAD_WEIGHTS
                        Load only weights from previously saved checkpoint
  --save-checkpoints-to SAVE_CHECKPOINTS_TO
                        Location where checkpoints will be stored
  --epochs EPOCHS       Number of epochs during training
  --batch-size BATCH_SIZE
                        Size of a single batch during training per GPU.
  --base-learning-rate BASE_LEARNING_RATE
                        Starting value of learning rate that might be changed
                        during training according to learning rate schedule
                        that is usually defined in detailed training
                        configuration.
  --gpu-num GPU_NUM     Number of GPUs that will be used in training, 0 is for
                        CPU mode.
```

## eval.py
```bash
python ../tools/eval.py -h
usage: eval.py [-h] --test-ann-files TEST_ANN_FILES --test-data-roots
               TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS --save-metrics-to
               SAVE_METRICS_TO [--save-output-to SAVE_OUTPUT_TO]
               [--wider-dir WIDER_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --test-ann-files TEST_ANN_FILES
                        Comma-separated paths to test annotation files.
  --test-data-roots TEST_DATA_ROOTS
                        Comma-separated paths to test data folders.
  --load-weights LOAD_WEIGHTS
                        Load only weights from previously saved checkpoint
  --save-metrics-to SAVE_METRICS_TO
                        Location where evaluated metrics values will be stored
                        (yaml file).
  --save-output-to SAVE_OUTPUT_TO
                        Location where output images (with displayed result of
                        model work) will be stored.
  --wider-dir WIDER_DIR
                        Location of WiderFace dataset.
```

## export.py
```bash
python ../../tools/export.py -h
usage: export.py [-h] --load-weights LOAD_WEIGHTS --save-model-to
                 SAVE_MODEL_TO [--onnx] [--openvino]
                 [--openvino-input-format OPENVINO_INPUT_FORMAT]
                 [--openvino-mo-args OPENVINO_MO_ARGS]

optional arguments:
  -h, --help            show this help message and exit
  --load-weights LOAD_WEIGHTS
                        Load only weights from previously saved checkpoint
  --save-model-to SAVE_MODEL_TO
                        Location where exported model will be stored.
  --onnx                Enable onnx export.
  --openvino            Enable OpenVINO export.
  --openvino-input-format OPENVINO_INPUT_FORMAT
                        Format of an input image for OpenVINO exported model.
  --openvino-mo-args OPENVINO_MO_ARGS
                        Additional args to OpenVINO Model Optimizer
```

## quantize.py
TBD
