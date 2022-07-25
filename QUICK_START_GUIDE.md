# Quick Start Guide

## Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.8+
* for training on GPU: [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)
   
**Note:** If using CUDA, make sure you are using a proper driver version. To do so, use `ls -la /usr/local | grep cuda`. If necessary, [install CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux) and select it with `export CUDA_HOME=/usr/local/cuda-11.1`.


## Setup OpenVINO™ Training Extensions

1. Clone the training_extensions repository with the following commands:
    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout develop origin/develop
    git submodule update --init --recursive
    ```

2. Install prerequisites with:
   ```
   sudo apt-get install python3-pip python3-venv
   ```

   Although they are not required, You may also want to use Jupyter notebooks or OTE CLI tools: 
   ```
   pip3 install notebook; cd ote_cli/notebooks/; jupyter notebook
   ```

3. Search for available scripts that create python virtual environments for different task types:
   ```bash
   find external/ -name init_venv.sh
   ```

   Sample output:
   ```
   external/mmdetection/init_venv.sh
   external/mmsegmentation/init_venv.sh
   external/deep-object-reid/init_venv.sh
   ```
   Each line in the output gives an `init_venv.sh` script that creates a virtual environment
   for the corresponding task type.

4. Choose a task type, for example,`external/mmdetection` for Object Detection.
   ```bash
   TASK_ALGO_DIR=./external/mmdetection/
   ```
   Note that the variable `TASK_ALGO_DIR` is set in this example for simplicity and will not be used in scripts.

5. Create and activate a virtual environment for the chosen task, then install the `ote_cli`.
   Note that the virtual environment directory may be created anywhere on your system.
   The `./cur_task_venv` is just an example used here for convenience.
   ```bash
   bash $TASK_ALGO_DIR/init_venv.sh ./cur_task_venv python3.8
   source ./cur_task_venv/bin/activate
   pip3 install -e ote_cli/ -c $TASK_ALGO_DIR/constraints.txt
   ```

   Note that `python3.8` is pointed as the second parameter of the script
   `init_venv.sh` -- it is the version of python that should be used. You can
   use any `python3.8+` version here if it is installed on your system.

   Also note that during installation of `ote_cli` the constraint file
   from the chosen task folder is used to avoid breaking constraints
   for the OTE task.

6. When `ote_cli` is installed in the virtual environment, you can use the
   `ote` command line interface to perform various actions for templates related to the chosen task type, such as running, training, evaluating, exporting, etc. 

## OTE CLI commands

### ote find 
   `ote find` lists model templates available for the given virtual environment.
   ```
   ote find --root $TASK_ALGO_DIR
   ```

   Output for the mmdetection used in the above example looks as follows:
   ```
   - id: Custom_Object_Detection_Gen3_VFNet
     name: VFNet
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_resnet50_VFNet/template.yaml
     task_type: DETECTION
   - id: Custom_Object_Detection_Gen3_ATSS
     name: ATSS
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml
     task_type: DETECTION
   - id: Custom_Object_Detection_Gen3_SSD
     name: SSD
     path: ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_SSD/template.yaml
     task_type: DETECTION
   - ...
   ```

### ote train 
   `ote train` trains a model (a particular model template) on a dataset and saves results in two files:
   * `weights.pth` - a model snapshot
   * `label_schema.json` - a label schema used in training, created from a dataset

   These files can be used by other `ote` commands: `ote export`, `ote eval`, `ote demo`.

  With the `--help` command, you can list additional information, such as its parameters common to all model templates and model-specific hyper parameters.

#### common parameters 
   command example:
   ```
   ote train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   output example:
   ```
   usage: ote train [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots
                    TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES
                    --val-data-roots VAL_DATA_ROOTS [--load-weights LOAD_WEIGHTS]
                    --save-model-to SAVE_MODEL_TO
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
     --save-model-to SAVE_MODEL_TO
                           Location where trained model will be stored.
     --enable-hpo          Execute hyper parameters optimization (HPO) before training.
     --hpo-time-ratio HPO_TIME_RATIO
                           Expected ratio of total time to run HPO to time taken for full fine-tuning.
   ```

#### model template-specific parameters 
   command example:
   ```
   ote train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml params --help
   ```

   output example:
   ```
   usage: ote train template params [-h]
                                    [--learning_parameters.batch_size BATCH_SIZE]
                                    [--learning_parameters.learning_rate LEARNING_RATE]
                                    [--learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS]
                                    [--learning_parameters.num_iters NUM_ITERS]
                                    [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD]
                                    [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD]
                                    [--nncf_optimization.enable_quantization ENABLE_QUANTIZATION]
                                    [--nncf_optimization.enable_pruning ENABLE_PRUNING]
                                    [--nncf_optimization.maximal_accuracy_degradation MAXIMAL_ACCURACY_DEGRADATION]
   
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
     --nncf_optimization.maximal_accuracy_degradation MAXIMAL_ACCURACY_DEGRADATION
                           header: Maximum accuracy degradation
                           type: FLOAT
                           default_value: 1.0
                           max_value: 100.0
                           min_value: 0.0
   ```

### ote optimize 
   `ote optimize` optimizes a pre-trained model using NNCF or POT depending on the model format.
   * NNCF optimization used for trained snapshots in a framework-specific format
   * POT optimization used for models exported in the OpenVINO IR format

   For example:
   Optimize a PyTorch model (.pth) with OpenVINO NNCF:
   ```
   ote optimize ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --load-weights weights.pth --save-model-to ./nncf_output --save-performance ./nncf_output/performance.json --train-ann-file ./data/car_tree_bug/annotations/instances_default.json --train-data-roots ./data/car_tree_bug/images --val-ann-file ./data/car_tree_bug/annotations/instances_default.json --val-data-roots ./data/car_tree_bug/images
   ```

   Optimize OpenVINO model (.bin or .xml) with OpenVINO POT:
   ```
   ote optimize ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --load-weights openvino.xml --save-model-to ./pot_output --save-performance ./pot_output/performance.json --train-ann-file ./data/car_tree_bug/annotations/instances_default.json --train-data-roots ./data/car_tree_bug/images --val-ann-file ./data/car_tree_bug/annotations/instances_default.json --val-data-roots ./data/car_tree_bug/images
   ```

   With the `--help` command, you can list additional information.
   command example:
   ```
   ote optimize ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Output example:
   ```
   usage: ote optimize [-h] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files
                    VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS --load-weights LOAD_WEIGHTS --save-model-to
                    SAVE_MODEL_TO [--aux-weights AUX_WEIGHTS]
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
                           Load weights of trained model
     --save-model-to SAVE_MODEL_TO
                           Location where trained model will be stored.
     --aux-weights AUX_WEIGHTS
                           Load weights of trained auxiliary model
   ```


### ote eval
   `ote eval` runs evaluation of a trained model on a particular dataset.

   With the `--help` command, you can list additional information, such as its parameters common to all model templates:
   command example:
   ```
   ote eval ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   output example:
   ```
   usage: ote eval [-h] --test-ann-files TEST_ANN_FILES --test-data-roots
                   TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS
                   [--save-performance SAVE_PERFORMANCE]
                   template {params} ...
   
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
     --save-performance SAVE_PERFORMANCE
                           Path to a json file where computed performance will be
                           stored.
   ```

### ote export 
   `ote export` exports a trained model to the OpenVINO format in order to efficiently run it on Intel hardware.
   
   With the `--help` command, you can list additional information, such as its parameters common to all model templates:
   command example:
   ```
   ote export ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   output example:
   ```
   usage: ote export [-h] --load-weights LOAD_WEIGHTS --save-model-to
                     SAVE_MODEL_TO
                     template
   
   positional arguments:
     template
   
   optional arguments:
     -h, --help            show this help message and exit
     --load-weights LOAD_WEIGHTS
                           Load only weights from previously saved checkpoint
     --save-model-to SAVE_MODEL_TO
                           Location where exported model will be stored.
   ```
### ote demo 
   `ote demo` runs model inference on images, videos, or webcam streams in order to see how it works with user's data
  
   With the `--help` command, you can list additional information, such as its parameters common to all model templates:
   command example:
   ```
   ote demo ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   output example:
   ```
   usage: ote demo [-h] -i INPUT --load-weights LOAD_WEIGHTS
                   [--fit-to-size FIT_TO_SIZE FIT_TO_SIZE] [--loop]
                   [--delay DELAY] [--display-perf]
                   template {params} ...
   
   positional arguments:
     template
     {params}              sub-command help
       params              Hyper parameters defined in template file.
   
   optional arguments:
     -h, --help            show this help message and exit
     -i INPUT, --input INPUT
                           Source of input data: images folder, image, webcam and
                           video.
     --load-weights LOAD_WEIGHTS
                           Load only weights from previously saved checkpoint
     --fit-to-size FIT_TO_SIZE FIT_TO_SIZE
                           Width and Height space-separated values. Fits
                           displayed images to window with specified Width and
                           Height. This options applies to result visualisation
                           only.
     --loop                Enable reading the input in a loop.
     --delay DELAY         Frame visualization time in ms.
     --display-perf        This option enables writing performance metrics on
                           displayed frame. These metrics take into account not
                           only model inference time, but also frame reading,
                           pre-processing and post-processing.
   ```

### ote deploy 
   `ote deploy` creates openvino.zip with a self-contained python package, a demo application, and an exported model. 
   
   With the `--help` command, you can list additional information, such as its parameters common to all model templates:
   command example:
   ```
   ote deploy ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   output example:
   ```
   usage: ote deploy [-h] --load-weights LOAD_WEIGHTS
                     [--save-model-to SAVE_MODEL_TO]
                     template
   
   positional arguments:
     template
   
   optional arguments:
     -h, --help            show this help message and exit
     --load-weights LOAD_WEIGHTS
                           Load only weights from previously saved checkpoint.
     --save-model-to SAVE_MODEL_TO
                           Location where openvino.zip will be stored.
   ```


---
\* Other names and brands may be claimed as the property of others.
