Object Detection model
======================

.. #TODO: Made Table of Concept for this page?

This tutorial reveals step by step procedure, how to install OTE CLI and train Object detection model on BBCD public dataset.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- `CUDA Toolkit 11.1 <https://developer.nvidia.com/cuda-11.1.1-download-archive>`_


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

3. Create and activate a virtual environment for the obect detection task, then install the ote_cli.
The following example shows that creating virtual environment to the ``det_venv`` folder in your current directory for detection task.

.. code-block::

    # create virtual env
    bash ./otx/algorithms/detection/init_venv.sh det_venv
    # activate virtual env
    source det_venv/bin/activate


***************************
Dataset preparation
***************************

1. Download a public `BCCD dataset <https://public.roboflow.com/object-detection/bccd/3>`_ (login required). Log in, put ``Download`` button and chose ``Terminal`` option. You will get the code line like this, but with your personal API key.

.. code-block::

  curl -L "https://public.roboflow.com/ds/<YOUR_API_KEY>" > bccd.zip;
  unzip bccd.zip; rm bccd.zip

#TODO insert data Sample

2. Check the file structure of downloaded dataset, which should be as following.

.. code-block::

  BCCD
  ├── README.dataset.txt
  ├── README.roboflow.txt
  ├── test/
      ├── _annotations.coco.json
      └── <images>
  ├──train/
      ├──_annotations.coco.json
      └── <images>
  └──valid/
      ├──_annotations.coco.json
      └── <images>


*********
Training
*********

1. Before training you need to chose, which object detection model will you use. The list of supported templates for object detection is available with the command line below. 

.. note::

  The characteristics and detailed comparison of the models could be found in Explanation section [#TODO link].

  To modify the arhitucture of supported models with various backbones check the Advanced tutorial for dataset modification [#TODO link].

.. code-block::

  (detection) ...$ otx find --template --task DETECTION
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
  |    TASK   |                 ID                |  NAME |                                    PATH                                   |
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
  | DETECTION |   Custom_Object_Detection_YOLOX   | YOLOX | otx/algorithms/detection/configs/detection/cspdarknet_yolox/template.yaml |
  | DETECTION |  Custom_Object_Detection_Gen3_SSD |  SSD  |  otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml |
  | DETECTION | Custom_Object_Detection_Gen3_ATSS |  ATSS | otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml |
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+

2. ``otx train`` trains a model (a particular model template) on a dataset and saves results in two files:

- weights.pth - a model snapshot
- label_schema.json - a label schema used in training, created from a dataset

These files can be used by other commands: ``export``, ``eval``, ``deploy`` and ``demo``.


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


3. For tutorial purposes, all examples will be run on the ATSS model. This comand line starts training of the medium object detection model on BCCD dataset.

.. code-block::

  (detection) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --train-ann-files ./BBCD/train/_annotations.coco.json 
                            --train-data-roots  ./BBCD/train 
                            --val-ann-files ./BBCD/valid/_annotations.coco.json 
                            --val-data-roots ./BBCD/valid 
                            --save-model-to ./outputs
                            --save-logs-to ./outputs/logs

To decrease batsch size or tune other trainig parameters, extend the comand line above with the following line.

.. code-block::

                            params --learning_parameters.batch_size 4 ...


The result of the training are weights.pth and label_schema.json, located in ``save-model-to`` folder.

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained model on a particular dataset.

Eval function receives test annotation information and folder that contains a model snapshot and label schema.

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


2. The command below evaluates snaphot in ``outputs`` folder on BCCD dataset and saves results to ``outputs/performance`` :

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --test-ann-files ./BBCD/train/_annotations.coco.json 
                            --test-data-roots  ./BBCD/train 
                            --load-weights ./outputs
                            --save-performance ./outputs/performance

  #TODO update metrics

3. #TODO ./outputs/performance output


*********
Export
*********
1. ``otx export`` exports a trained pth model to the OpenVINO format in order to efficiently run it on Intel hardware. Also, the resulting IR model is required to run POT optimization in section below.

2. The command below performs exporting to the trained model ``outputs/weights.pth`` in previous section and save exported model to the ``outputs/openvino/`` folder.

.. code-block::

  (detection) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                              --load-weights outputs/weights.pth
                              --save-model-to outputs/openvino/

  #TODO update output


*************
Optimization
*************

1. ``otx optimize`` optimizes a model using NNCF or POT depending on the model format.

- NNCF optimization used for trained snapshots in a framework-specific format such as checkpoint (pth) file from Pytorch. It optimizes model during training.
- POT optimization used for models exported in the OpenVINO IR format. It performs post-training optimization.

2. Command example for optimizing a PyTorch model (.pth) with OpenVINO NNCF.

.. code-block::

  (detection) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
                                --train-ann-files BBCD/train/_annotations.coco.json 
                                --train-data-roots  BBCD/train 
                                --val-ann-files BBCD/valid/_annotations.coco.json 
                                --val-data-roots BBCD/valid 
                                --load-weights outputs/weights.pth
                                --save-model-to outputs/nncf
                                --save-performance outputs/nncf/performance.json
  #TODO update metrics

3. Command example for optimizing OpenVINO model (.xml) with OpenVINO POT:

.. code-block::

  (detection) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
                                --train-ann-files BBCD/train/_annotations.coco.json 
                                --train-data-roots  BBCD/train 
                                --val-ann-files BBCD/valid/_annotations.coco.json 
                                --val-data-roots BBCD/valid 
                                --load-weights outputs/weights.pth
                                --save-model-to outputs/pot
                                --save-performance outputs/pot/performance.json
  #TODO update metrics


The following stages how to deploy model and run demo are described in [link].

