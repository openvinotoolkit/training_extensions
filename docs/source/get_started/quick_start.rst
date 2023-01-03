###########
Quick Start
###########

<Guidelines for writing>

Quick Start describes the most common instructions with the default sample datasets and 
default model template in a task (e.g. classification task). It often accompany those instructions 
with easy-to-understand pictures.

It can contain 1) model training 2) model evalaution 3) model inference 4) model export 5) model deploy


##################################
Extra code that can be useful here
##################################

************************************
Setup OpenVINO™ Training Extensions
************************************

1. Clone the training_extensions repository with the following commands:

.. code-block::

    git clone https://github.com/openvinotoolkit/training_extensions.git
    cd training_extensions
    git checkout develop

2. Install prerequisites with:

.. code-block::

    sudo apt-get install python3-pip python3-venv
    # verify your python version
    python3 --version; pip3 --version; 


Output should be similar to that.

.. code-block::
  
    Python 3.8.10
    pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)

*********
Training
*********

1. ``otx train`` trains a model (a particular model template) on a dataset and saves results in two files:

- weights.pth - a model snapshot
- label_schema.json - a label schema used in training, created from a dataset

These files can be used by other commands: ``export``, ``eval``, ``deploy`` and ``demo``.


==================
Сommon parameters
==================

Train function receives common parameters regardless of the task (for classification, detection, etc), which contains annotation information, unlabeled annotation for semi-supervised learning, where to save model and HPO (Hyperparameter optimization) [link?] enablement.

.. code-block::

  (detection)...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml --help

  usage: otx train [-h] [--data DATA] --train-ann-files TRAIN_ANN_FILES --train-data-roots TRAIN_DATA_ROOTS --val-ann-files VAL_ANN_FILES --val-data-roots VAL_DATA_ROOTS [--unlabeled-data-roots UNLABELED_DATA_ROOTS] [--unlabeled-file-list UNLABELED_FILE_LIST] [--load-weights LOAD_WEIGHTS]
                  [--save-model-to SAVE_MODEL_TO] [--save-logs-to SAVE_LOGS_TO] [--enable-hpo] [--hpo-time-ratio HPO_TIME_RATIO]
                  template {params} ...

  positional arguments:
    template
    {params}              sub-command help
      params              Hyper parameters defined in template file.

  optional arguments:
    -h, --help            show this help message and exitfnxhn
    --data DATA
    --train-ann-files TRAIN_ANN_FILES
                          Comma-separated paths to training annotation files.
    --train-data-roots TRAIN_DATA_ROOTS
                          Comma-separated paths to training data folders.
    --val-ann-files VAL_ANN_FILES
                          Comma-separated paths to validation annotation files.
    --val-data-roots VAL_DATA_ROOTS
                          Comma-separated paths to validation data folders.
    --unlabeled-data-roots UNLABELED_DATA_ROOTS
                          Comma-separated paths to unlabeled data folders
    --unlabeled-file-list UNLABELED_FILE_LIST
                          Comma-separated paths to unlabeled file list
    --load-weights LOAD_WEIGHTS
                          Load only weights from previously saved checkpoint
    --save-model-to SAVE_MODEL_TO
                          Location where trained model will be stored.
    --save-logs-to SAVE_LOGS_TO
                          Location where logs will be stored.
    --enable-hpo          Execute hyper parameters optimization (HPO) before training.
    --hpo-time-ratio HPO_TIME_RATIO
                          Expected ratio of total time to run HPO to time taken for full fine-tuning.

============================
Template-specific parameters
============================

In order to tune training parameters such as batch size, learning rate, a various set of parameters can be updated via comand line.

.. code-block::

  otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml params --help
  usage: otx train template params [-h] [--learning_parameters.batch_size BATCH_SIZE] [--learning_parameters.learning_rate LEARNING_RATE] [--learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS]
                                  [--learning_parameters.num_iters NUM_ITERS] [--learning_parameters.enable_early_stopping ENABLE_EARLY_STOPPING] [--learning_parameters.early_stop_start EARLY_STOP_START]
                                  [--learning_parameters.early_stop_patience EARLY_STOP_PATIENCE] [--learning_parameters.early_stop_iteration_patience EARLY_STOP_ITERATION_PATIENCE]
                                  [--learning_parameters.use_adaptive_interval USE_ADAPTIVE_INTERVAL] [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD]
                                  [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD] [--algo_backend.train_type TRAIN_TYPE] [--nncf_optimization.enable_quantization ENABLE_QUANTIZATION]
                                  [--nncf_optimization.enable_pruning ENABLE_PRUNING] [--nncf_optimization.pruning_supported PRUNING_SUPPORTED]

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
                          default_value: 0.004
                          max_value: 0.1
                          min_value: 1e-07
    --learning_parameters.learning_rate_warmup_iters LEARNING_RATE_WARMUP_ITERS
                          header: Number of iterations for learning rate warmup
                          type: INTEGER
                          default_value: 3
                          max_value: 10000
                          min_value: 0

    ...


***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained model on a particular dataset.

Eval function receives test annotation information and folder containig a model snapshot and label schema.

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml --help
  usage: otx eval [-h] [--data DATA] --test-ann-files TEST_ANN_FILES --test-data-roots TEST_DATA_ROOTS --load-weights LOAD_WEIGHTS [--save-performance SAVE_PERFORMANCE] template {params} ...

  positional arguments:
    template
    {params}              sub-command help
      params              Hyper parameters defined in template file.

  optional arguments:
    -h, --help            show this help message and exit
    --data DATA
    --test-ann-files TEST_ANN_FILES
                          Comma-separated paths to test annotation files.
    --test-data-roots TEST_DATA_ROOTS
                          Comma-separated paths to test data folders.
    --load-weights LOAD_WEIGHTS
                          Load only weights from previously saved checkpoint
    --save-performance SAVE_PERFORMANCE
                          Path to a json file where computed performance will be stored.

The default metric measured is mAP and f1.

In order to tune testing parameters such as confidence threshold, a various set of parameters can be updated via comand line.

.. code-block:: 

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml params --help
  usage: otx eval template params [-h] [--postprocessing.confidence_threshold CONFIDENCE_THRESHOLD] [--postprocessing.result_based_confidence_threshold RESULT_BASED_CONFIDENCE_THRESHOLD]
                                  [--nncf_optimization.enable_quantization ENABLE_QUANTIZATION] [--nncf_optimization.enable_pruning ENABLE_PRUNING]

  optional arguments:
    -h, --help            show this help message and exit
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
    ...


*********
Export
*********
1. ``otx export`` exports a trained pth model to the OpenVINO format in order to efficiently run it on Intel hardware. Also, the resulting IR model is required to run POT optimization in section below.

2. The command below performs exporting of the ``outputs/weights.pth`` trained in previous section and saves exported model to the ``outputs/openvino/`` folder.

.. code-block::

  (detection) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                              --load-weights outputs/weights.pth
                              --save-model-to outputs/openvino/



*************
Optimization
*************

1. ``otx optimize`` optimizes a model using NNCF or POT depending on the model format, using NNCF framework [#TODO link].

- NNCF optimization used for trained snapshots in a framework-specific format such as checkpoint (pth) file from Pytorch. It optimizes model during training.
- POT optimization used for models exported in the OpenVINO IR format. It performs post-training optimization.

2. Command example for optimizing a PyTorch model (.pth) with OpenVINO NNCF.

3. Command example for optimizing OpenVINO model (.xml) with OpenVINO POT:
