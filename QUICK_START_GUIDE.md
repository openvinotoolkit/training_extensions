# Quick Start Guide

## Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.8+
* [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) - for training on GPU

## Setup OpenVINO™ Training Extensions

1. Clone repository in the working directory by running the following:
    ```
    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
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

   Sample output:
   ```
   external/mmdetection/init_venv.sh
   external/mmsegmentation/init_venv.sh
   external/deep-object-reid/init_venv.sh
   ```
   Each line in the output gives an `init_venv.sh` script that creates a virtual environment
   for the corresponding task type.

4. Let's choose a task type.
   Let it be `external/mmdetection` for Object Detection task.
   ```bash
   TASK_ALGO_DIR=./external/mmdetection/
   ```
   Note that we will not use the variable `TASK_ALGO_DIR` inside our scripts, we set it just to
   simplify this guide.

5. Let's create, activate virtual environment for the chosen task, and install `ote_cli`.
   Note that the virtual environment folder may be created in any place in your system,
   but we will create it in the folder `./cur_task_venv` for convenience.
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

6. As soon as `ote_cli` is installed in the virtual environment, you can use
   `ote` command line interface described below to run
   train/eval/export/other action for templates related to the chosen task type.

## OTE CLI commands

### ote find - search for model templates
   Have a look at model templates available for this virtual environment:
   ```
   ote find --root $TASK_ALGO_DIR
   ```

   Sample output (for mmdetection task chosen above):
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
   Let's choose `./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml`

### ote train - run training of a particular model template
   This tool trains a model on a dataset and saves results as following artifacts:
   * `weights.pth` - a model snapshot
   * `label_schema.json` - a label schema used in training, created from a dataset

   These artifacts can be used by other `ote` commands: `ote export`, `ote eval`, `ote demo`.
   Let's have a look at `ote train` help. These parameters are the same for all model templates.
   ```
   ote train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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

   Let's have a look at `ote train` hyper parameters help. These parameters are model template specific.
   ```
   ote train ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml params --help
   ```

   Sample output:
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

### ote optimize - run optimization of a particular model template
   This tool optimizes a trained model using NNCF or POT depending on a model format:

   - NNCF optimization used for trained snapshots in framework specific format
   - POT optimization used for exported model in IR format

   ```
   ote optimize ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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


### ote eval - run evaluation of a trained model on particular dataset
   Let's have a look at `ote eval` help. These parameters are the same for all model templates.
   ```
   ote eval ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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

### ote export - export a trained model to the OpenVINO format in order to efficiently run it on Intel hardware
   Let's have a look at `ote export` help. These parameters are the same for all model templates.
   ```
   ote export ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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
### ote demo - run model inference on images, videos, webcam in order to see how it works on user's data
   Let's have a look at `ote demo` help. These parameters are the same for all model templates.
   ```
   ote demo ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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

### ote deploy - create openvino.zip with self-contained python package, demo application and exported model
   Let's have a look at `ote deploy` help. These parameters are the same for all model templates.
   ```
   ote deploy ./external/mmdetection/configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/template.yaml --help
   ```

   Sample output:
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

## OTE Jupyter Nootebooks
One can use Jupyter notebooks or OTE CLI tools to start working with models:
```
pip3 install notebook; cd ote_cli/notebooks/; jupyter notebook
```

---
\* Other names and brands may be claimed as the property of others.
