Object Detection model
======================

#TODO: Made Table of Content for this page?

This tutorial reveals end-to-end solution from installation to model deploying for object detection task. We show how to train, validate, export and optimize ATSS model on BBCD public dataset.
To learn how to deploy the trained model refer to deploy tutorial [#TODO link].
To learn, how to run the demo, refer to the demo tutorial [#TODO link].

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- CUDA Toolkit 11.4 



*************************
Setup virtual environment
*************************

You can follow the installation process from a quick_start quide [#TODO link] to create a univesal virtual environment for all tasks. On other hand, to save memory and time, you can create task-specific environment following the process below.

1. Install prerequisites with:

.. code-block::

    sudo apt-get install python3-pip python3-venv
    # verify your python version
    python3 --version; pip3 --version; 

    Python 3.8.10
    pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)

2. Create and activate a virtual environment for the obect detection task, then install the ote_cli.
The following example shows that creating virtual environment to the ``det_venv`` folder in your current directory for detection task.

.. code-block::

    # create virtual env
    bash ./otx/algorithms/detection/init_venv.sh det_venv
    # activate virtual env
    source det_venv/bin/activate


***************************
Dataset preparation
***************************

1. Download a public `BCCD dataset <https://public.roboflow.com/object-detection/bccd/3>`_ (login required). Log in, click ``Download`` button and chose ``Terminal`` option. You will get the code line like this, but with your personal API key.

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


3. ``(Optional)`` To simplify the comand line functions calling, you may create a ``data.yaml`` file with annotations info and pass it as a parameter. The content of the ``training_extesions/data.yaml`` for BBCD dataset should have absolete paths and will be similar to that:

.. code-block::

  {'data': 
    {'train': 
      {'ann-files': '/home/gzalessk/training_extensions/BBCD/train/_annotations.coco.json',
       'data-roots': '/home/gzalessk/training_extensions/datasets/BBCD/train'},
    'val':
      {'ann-files': '/home/gzalessk/training_extensions/datasets/BBCD/valid/_annotations.coco.json',
       'data-roots': '/home/gzalessk/training_extensions/datasets/BBCD/valid'},
    'test':
      {'ann-files': '/home/gzalessk/training_extensions/datasets/BBCD/test/_annotations.coco.json',
       'data-roots': '/home/gzalessk/training_extensions/datasets/BBCD/test'}
    }
  }


*********
Training
*********

1. Before training you need to chose, which object detection model will you use. The list of supported templates for object detection is available with the command line below. 

.. note::

  The characteristics and detailed comparison of the models could be found in Explanation section [#TODO link].

  To modify the architecture of supported models with various backbones, please refer to the advanced tutorial for model customization  [#TODO link].

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


3. For tutorial purposes, all examples will be run on the ATSS model. This comand line starts 1 GPU training of the medium object detection model on BCCD dataset.

.. code-block::

  (detection) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --train-ann-files BBCD/train/_annotations.coco.json 
                            --train-data-roots  BBCD/train 
                            --val-ann-files BBCD/valid/_annotations.coco.json 
                            --val-data-roots BBCD/valid 
                            --save-model-to outputs
                            --work-dir outputs/logs
                            --gpus 1

If you created ``data.yaml`` file in previous step, you can simplify the training by specifying it as a ``data`` parameter:

.. code-block::

  (detection) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml
                            --save-model-to outputs
                            --work-dir outputs/logs
                            --gpus 1


Additionally, you can tune training parameters such as batch size, learning rate, patience epochs or warm-up iteration. You can read more about template-specific parameters in quick start [#TODO link].
It can be done by manual updating parameters in ``template.yaml`` file or via comand line. 

For example, to decrease batsch size to 4, fix the number of epochs to 100 and disable early stopping, extend the comand line above with the following line.

.. code-block::

                            params --learning_parameters.batch_size 4 --learning_parameters.num_iters 100 --learning_parameters.enable_early_stopping false 


The result of the training are ``weights.pth`` and ``label_schema.json``, located in ``outputs`` folder, and logs in the ``outputs/logs`` dir.

.. code-block::

  ...

  2022-12-29 00:59:51,961 - mmdet - INFO - workflow: [('train', 1)], max: 200 epochs
  [ INFO ] workflow: [('train', 1)], max: 200 epochs
  [ INFO ]  workflow: %s, max: %d epochs
  2022-12-29 00:59:51,965 | INFO : cancel hook is initialized
  2022-12-29 00:59:51,965 | INFO : logger in the runner is replaced to the MPA logger
  2022-12-29 00:59:51,975 | INFO : Update Lr patience: 3
  2022-12-29 00:59:51,975 | INFO : Update Validation Interval: 2
  2022-12-29 00:59:51,975 | INFO : Update Early-Stop patience: 5
  2022-12-29 01:00:30,180 | INFO : Epoch [1][1/32]        lr: 1.333e-03, eta: 282 days, 22:46:42, time: 38.204, data_time: 0.462, memory: 4669, current_iters: 0, loss_cls: 1.1113, loss_bbox: 0.5567, loss_centerness: 0.5920, loss: 2.2600, grad_norm: 3.6441

  ...
  ---------------iou_thr: 0.5---------------

  +-----------+-----+------+--------+-------+
  | class     | gts | dets | recall | ap    |
  +-----------+-----+------+--------+-------+
  | Platelets | 76  | 310  | 1.000  | 0.897 |
  | RBC       | 819 | 4070 | 0.994  | 0.903 |
  | WBC       | 72  | 516  | 1.000  | 0.988 |
  +-----------+-----+------+--------+-------+
  | mAP       |     |      |        | 0.929 |
  +-----------+-----+------+--------+-------+
  2022-12-29 01:08:52,434 | INFO : run task done.
  2022-12-29 01:08:53,010 | INFO : Adjusting the confidence threshold
  2022-12-29 01:08:53,520 | INFO : Setting confidence threshold to 0.32500000000000007 based on results
  2022-12-29 01:08:53,521 | INFO : Final model performance: Performance(score: 0.8315842078960519, dashboard: (17 metric groups))


***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained model on a particular dataset.

Eval function receives test annotation information and model snapshot, trained in previous step.
Please note, that ``label schema.json`` file should be located in the same folder with model snaphot, as it contains meta information about the dataset .

The default metric measured is F1 measure.

2. The command below evaluates snaphot in ``outputs`` folder on BCCD dataset and saves results to ``outputs/performance`` :

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --test-ann-files BBCD/valid/_annotations.coco.json 
                            --test-data-roots  BBCD/valid 
                            --load-weights outputs/weights.pth
                            --save-performance outputs/performance.json
  

If ``data.yaml`` was created, the command can be simplified by passing it for a ``--data`` parameter. Note, that this line will run validation on the test set (not validation set):

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml 
                            --load-weights outputs/weights.pth
                            --save-performance outputs/performance.json

The validation output will look as following:

.. code-block::

  2022-12-29 01:32:00,505 | INFO : run task done.
  2022-12-29 01:32:01,215 | INFO : Inference completed
  2022-12-29 01:32:01,216 | INFO : called evaluate()
  2022-12-29 01:32:01,527 | INFO : F-measure after evaluation: 0.8315842078960519



Additionally, you can tune testing parameters such as confidence threshold via comand line. You can read more about template-specific parameters for validation in quick start [#TODO link].
For example, to increase the confidence treshold to decrease the number of False Positive predictions (there you have prediction, but don't have annotated object for it) update the evaluation comand line as it's shown below. 
Please note, that by default confidence treshold is detected automatically based on result to maximize final F1 metric. So, to set custom confidence trashold, please disable ``result_based_confidence_threshold`` option.

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml 
                            --load-weights outputs/weights.pth
                            params 
                            --postprocessing.confidence_threshold 0.5
                            --postprocessing.result_based_confidence_threshold false 

...

2023-01-03 18:55:01,956 | INFO : F-measure after evaluation: 0.6274238227146813

3. The output of ``./outputs/performance.json`` consists of dict with target metric name and its value.

.. code-block::

  {"f-measure": 0.8315842078960519}


*********
Export
*********
1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format in order to efficiently run it on Intel hardware. Also, the resulting IR model is required to run POT optimization in the section below.

2. The command below performs exporting of the trained model ``outputs/weights.pth`` in previous section and saves the exported model to the ``outputs/openvino/`` folder.

.. code-block::

  (detection) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                              --load-weights outputs/weights.pth
                              --save-model-to outputs/openvino/

  ...

  2022-12-29 01:39:11,980 | INFO : Exporting completed
  2022-12-29 01:39:11,980 | INFO : run task done.
  2022-12-29 01:39:11,990 | INFO : Exporting completed


3. You can check the accuracy of exported model as simple as accuracy of the ``.pth`` model, using ``otx eval`` with the path of IR model.

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --test-ann-files BBCD/valid/_annotations.coco.json 
                            --test-data-roots  BBCD/valid 
                            --load-weights outputs/openvino/openvino.xml
                            --save-performance outputs/openvino/performance.json
  
  ...



*************
Optimization
*************

1. ``otx optimize`` optimizes a model using NNCF or POT depending on the model format.

- NNCF optimization is used for trained snapshots in a framework-specific format such as checkpoint (pth) file from Pytorch. It starts training-aware quantization based on the obtained weights from the training stage.
- POT optimization is used for models exported in the OpenVINO™ IR format. It decreases floating-point precision to integer precision of the exported model by performing the post-training optimization.

The function results with a following files, which could be used to run ``otx demo``[link]:

- confidence_threshold
- config.json
- label_schema.json
- openvino.bin
- openvino.xml

Read more about optimization in [#TODO link]

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

  ...

  2022-12-29 02:11:49,018 | INFO : Loaded model weights from Task Environment
  2022-12-29 02:11:49,018 | INFO : Model architecture: ATSS
  2022-12-29 02:11:49,018 | INFO : Loaded model weights from Task Environment
  2022-12-29 02:11:49,018 | INFO : Model architecture: ATSS
  2022-12-29 02:11:49,019 | INFO : Task initialization completed
  INFO:nncf:Please, provide execution parameters for optimal model initialization
  2022-12-29 02:11:56,996 - mmdet - INFO - Received non-NNCF checkpoint to start training -- initialization of NNCF fields will be done
  [ INFO ] Received non-NNCF checkpoint to start training -- initialization of NNCF fields will be done
  [ INFO ]  Received non-NNCF checkpoint to start training -- initialization of NNCF fields will be done
  2022-12-29 02:11:56,999 - mmdet - INFO - Calculating an original model accuracy
  ...

  INFO:nncf:Original model accuracy: 0.4319
  INFO:nncf:Compressed model accuracy: 0.5564
  INFO:nncf:Absolute accuracy drop: -0.1245
  INFO:nncf:Relative accuracy drop: -28.82%
  INFO:nncf:Accuracy budget: 0.1345


#TODO significant drop of the loaded snapshot: 0.43 instead of 0.83
#TODO The optimized model isn't being saved (TypeError: cannot pickle '_thread.lock' object)
#TODO rebase on feature/otx once NNCF will be fixed

3. Command example for optimizing OpenVINO model (.xml) with OpenVINO POT:

.. code-block::

  (detection) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
                                --train-ann-files BBCD/train/_annotations.coco.json 
                                --train-data-roots  BBCD/train 
                                --val-ann-files BBCD/valid/_annotations.coco.json 
                                --val-data-roots BBCD/valid 
                                --load-weights outputs/openvino/openvino.xml
                                --save-model-to outputs/pot

  ...

  2022-12-31 05:31:04,125 | INFO : Loading OpenVINO OTXDetectionTask
  2022-12-31 05:31:05,470 | INFO : OpenVINO task initialization completed
  2022-12-31 05:31:05,470 | INFO : Start POT optimization

  ...

  2022-12-31 05:37:51,004 | INFO : POT optimization completed
  2022-12-31 05:37:51,219 | INFO : Start OpenVINO inference
  2022-12-31 05:37:55,423 | INFO : OpenVINO inference completed
  2022-12-31 05:37:55,423 | INFO : Start OpenVINO metric evaluation
  2022-12-31 05:37:55,776 | INFO : OpenVINO metric evaluation completed
  Performance(score: 0.8343621399176954, dashboard: (1 metric groups))

The POT optimization will take 5-10 minutes without logging.


The following stages how to deploy model and run demo are described in [link].

4. You can evaluate the optimized model passing it to ``otx eval`` function.

***************
Troubleshooting
***************

#TODO possible error logs and their solution?
