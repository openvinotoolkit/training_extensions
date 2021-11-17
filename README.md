# OpenVINO™ Training Extensions

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Quick Start Guide

### Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.8+
* [OpenVINO™](https://software.intel.com/en-us/openvino-toolkit) - for exporting and running models
* [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) - for training on GPU

### Setup OpenVINO™ Training Extensions

1. Clone repository in the working directory by running the following:
    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout -b develop origin/develop
    git submodule update --init --recursive
    ```

2. Install prerequisites by running the following:
    ```
    sudo apt-get install python3-pip python3-venv
    ```

3. Search for available scripts that create python virtual environments for different task types:
   ```bash
   find external/ -name init_venv.sh
   ```
   
   Approximate output:
   ```
   external/mmdetection/init_venv.sh
   external/mmsegmentation/init_venv.sh
   external/deep-object-reid/init_venv.sh
   ```
   
4. Let's create, activate Object Detection virtual environment and install `ote_cli`:
   ```
   ./external/mmdetection/init_venv.sh det_venv
   source det_venv/bin/activate
   pip3 install -e ote_cli/
   ```
   
### OTE CLI tools
1. Have a look at model templates available for this virtual environment:
   ```
   ote_find --root ./external/mmdetection/
   ```
   
   Approximate output:
   ```
   - framework: OTEDetection_v2.9.1
     name: Custom_Object_Detection_Gen3_VFNet
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_resnet50_VFNet/template.yaml
     task_type: DETECTION
   - framework: OTEDetection_v2.9.1
     name: Custom_Object_Detection_Gen3_SSD
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_SSD/template.yaml
     task_type: DETECTION
   - framework: OTEDetection_v2.9.1
     name: Custom_Object_Detection_Gen3_ATSS
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml
     task_type: DETECTION
   - ...
   ```
   Let's choose `./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml`

2. Let's have a look at `ote_train` help. These parameters are the same for all model templates.
   ```
   ote_train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```
   
   Approximate output:
   ```
   ote_train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   usage: ote_train [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS [--load-weights    LOAD_WEIGHTS] --save-weights
                    SAVE_WEIGHTS
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
     --save-weights SAVE_WEIGHTS
                           Location to store wiehgts.
   ```
                        
3. Let's have a look at `ote_train` hyper parameters help. These parameters are model template specific.
   ```
   ote_train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml params --help
   ```
   
   Approximate output:
   ```
   usage: ote_train template params [-h] [--learning_parameters.batch_size BATCH_SIZE] [--learning_parameters.learning_rate LEARNING_RATE]
                                   [--learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS] [--learning_parameters.num_iters NUM_ITERS]
                                   [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD] [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD]
                                   [--nncf_optimization.enable_quantization ENABLE_QUANTIZATION] [--nncf_optimization.enable_pruning ENABLE_PRUNING]
    
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
                         default_value: 0.008
                         max_value: 0.1
                         min_value: 1e-07
   --learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS
                         header: Number of iterations for learning rate warmup
                         type: INTEGER
                         default_value: 200
                         max_value: 10000
                         min_value: 0
   --learning_parameters.num_iters NUM_ITERS
                         header: Number of training iterations
                         type: INTEGER
                         default_value: 300
                         max_value: 100000
                         min_value: 1
   --postprocessing.confidence_threshold CONFIDENCE_THRESHOLD
                         header: Confidence threshold
                         type: FLOAT
                         default_value: 0.35
                         max_value: 1
                         min_value: 0
   --postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD
                         header: Result based confidence threshold
                         type: BOOLEAN
                         default_value: True
   --nncf_optimization.enable_quantization ENABLE_QUANTIZATION
                         header: Enable quantization algorithm
                         type: BOOLEAN
                         default_value: True
   --nncf_optimization.enable_pruning ENABLE_PRUNING
                         header: Enable filter pruning algorithm
                         type: BOOLEAN
                         default_value: False
   ```
4. Let's have a look at `ote_eval` help. These parameters are the same for all model templates.
   ```
   ote_eval ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```
   
   Approximate output:
   ```
   usage: ote_eval [-h] --test-ann-files TEST_ANN_FILES --test-data-roots TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS template {params} ...
    
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
                           Load only weights from previously saved checkpoint
   ```

5. Let's have a look at `ote_export` help. These parameters are the same for all model templates.
   ```
   ote_export ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```
   
   Approximate output:
   ```
   usage: ote_export [-h] --load-weights LOAD_WEIGHTS --save-model-to SAVE_MODEL_TO [--ann-files ANN_FILES] [--labels LABELS [LABELS ...]] template

   positional arguments:
     template

   optional arguments:
     -h, --help            show this help message and exit
     --load-weights LOAD_WEIGHTS
                           Load only weights from previously saved checkpoint
     --save-model-to SAVE_MODEL_TO
                           Location where exported model will be stored.
     --ann-files ANN_FILES
     --labels LABELS [LABELS ...]
   ```

### OTE Jupyter Nootebooks
One can use Jupyter notebooks or OTE CLI tools to start working with models:
```
pip3 install notebook; cd ote_cli/notebooks/; jupyter notebook
```
  
## Misc

Models that were previously developed can be found [here](misc).

## Contributing

Please read the [contribution guidelines](CONTRIBUTING.md) before starting work on a pull request.

## Known Limitations

Currently, training, exporting, evaluation scripts for TensorFlow\*-based models and the most of PyTorch\*-based models from [Misc](#misc) section are exploratory and are not validated.

---
\* Other names and brands may be claimed as the property of others.
