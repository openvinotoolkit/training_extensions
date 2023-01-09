Object Detection model
======================

This tutorial reveals end-to-end solution from installation to model deploying for object detection task on a certain example.
On this page we show how to train, validate, export and optimize ATSS model on BCCD public dataset.

To learn how to deploy the trained model refer to: :doc:`../deploy`.

To learn, how to run the demo and visualize results, refer to: :doc:`../demo`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- CUDA Toolkit 11.4 



*************************
Setup virtual environment
*************************

You can follow the installation process from a :doc:`quick_start quide <../../../get_started/quick_start>` to create a univesal virtual environment for all tasks. On other hand, to save memory and time, you can create task-specific environment following the process below.

1. Install prerequisites with:

.. code-block::

    sudo apt-get install python3-pip python3-venv
    # verify your python version
    python3 --version; pip3 --version; 

    Python 3.8.10
    pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)

2. Create and activate a virtual environment for the obect detection task.
The following example creates a virtual environment in the ``det_venv`` folder for detection task.

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

This dataset contains images of blood cells. It's a great example to start with, because the training model achieves high accuracy right grom the beginning due to large and focused objects.

.. image:: ../../../../utils/images/bccd_sample_image.jpg
  :width: 600
  :alt: this image uploaded from test set of this `source <https://public.roboflow.com/object-detection/bccd/3>`_


2. Check the file structure of downloaded dataset, which should look like this.

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


3. ``(Optional)`` To simplify the command line functions calling, we may create a ``data.yaml`` file with annotations info and pass it as a ``--data`` parameter. The content of the ``training_extesions/data.yaml`` for BCCD dataset should have absolete paths and will be similar to that:

.. code-block::

  {'data': 
    {'train': 
      {'ann-files': '/home/<username>/training_extensions/BCCD/train/_annotations.coco.json',
       'data-roots': '/home/<username>/training_extensions/datasets/BCCD/train'},
    'val':
      {'ann-files': '/home/<username>/training_extensions/datasets/BCCD/valid/_annotations.coco.json',
       'data-roots': '/home/<username>/training_extensions/datasets/BCCD/valid'},
    'test':
      {'ann-files': '/home/<username>/training_extensions/datasets/BCCD/valid/_annotations.coco.json',
       'data-roots': '/home/<username>/training_extensions/datasets/BCCD/valid'}
    }
  }


*********
Training
*********

1. First of all, we need to chose, which object detection model will we train. The list of supported templates for object detection is available with the command line below. 

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/Main_algorithms/object_detection>`.

  To modify the architecture of supported models with various backbones, please refer to the :doc:`advanced tutorial for model customization <../../advanced/backbones>`.

.. code-block::

  (detection) ...$ otx find --template --task DETECTION
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
  |    TASK   |                 ID                |  NAME |                                    PATH                                   |
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+
  | DETECTION |   Custom_Object_Detection_YOLOX   | YOLOX | otx/algorithms/detection/configs/detection/cspdarknet_yolox/template.yaml |
  | DETECTION |  Custom_Object_Detection_Gen3_SSD |  SSD  |  otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml |
  | DETECTION | Custom_Object_Detection_Gen3_ATSS |  ATSS | otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml |
  +-----------+-----------------------------------+-------+---------------------------------------------------------------------------+

2. ``otx train`` trains a model (a particular model template) on a dataset and results in two files:

- ``weights.pth`` - a model snapshot
- ``label_schema.json`` - a label schema used in training, created from a dataset

These are needed as inputs for the further commands: ``export``, ``eval``,  ``optimize``,  ``deploy`` and ``demo``.


3. To have a specific example in this tutorial, all commands will be run on the ATSS model. For instance, this command line starts 1 GPU training of the medium object detection model on BCCD dataset:

.. code-block::

  (detection) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --train-ann-files BCCD/train/_annotations.coco.json 
                            --train-data-roots  BCCD/train 
                            --val-ann-files BCCD/valid/_annotations.coco.json 
                            --val-data-roots BCCD/valid 
                            --save-model-to outputs
                            --work-dir outputs/logs
                            --gpus 1

If you created ``data.yaml`` file in previous step, you can simplify the training by passing it in ``--data`` parameter:

.. code-block::

  (detection) ...$ otx train otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml
                            --save-model-to outputs
                            --work-dir outputs/logs
                            --gpus 1

Looks much simplier, isn't it?

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iteration. More about template-specific parameters is in quick start [#TODO link].

It can be done by manual updating parameters in ``template.yaml`` file or via comand line. 

For example, to decrease batsch size to 4, fix the number of epochs to 100 and disable early stopping, extend the comand line above with the following line.

.. code-block::

                            params --learning_parameters.batch_size 4 --learning_parameters.num_iters 100 --learning_parameters.enable_early_stopping false 


5. The training results with ``weights.pth`` and ``label_schema.json`` files, located in ``outputs`` folder, while training logs can be found in the ``outputs/logs`` dir.

.. code-block::

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

Now we have the Pytorch object detection model trained with OTX, that we can use for evaliation, export, optimization and deployment. 

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained model on a particular dataset.

Eval function receives test annotation information and model snapshot, trained in previous step.
Please note, that ``label schema.json`` file should be located in the same folder with model snapshot, as it contains meta information about the dataset.

The default metric is F1 measure.

2. That's how we can evaluate the snaphot in ``outputs`` folder on BCCD dataset and save results to ``outputs/performance``:

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --test-ann-files BCCD/valid/_annotations.coco.json 
                            --test-data-roots  BCCD/valid 
                            --load-weights outputs/weights.pth
                            --save-performance outputs/performance.json
  

If you created ``data.yaml`` file in previous step, you can simplify the training by passing it in ``--data`` parameter. 
Note, that this line will run validation on the test set (not validation set):

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml 
                            --load-weights outputs/weights.pth
                            --save-performance outputs/performance.json

We will get this validation output:

.. code-block::

  2022-12-29 01:32:00,505 | INFO : run task done.
  2022-12-29 01:32:01,215 | INFO : Inference completed
  2022-12-29 01:32:01,216 | INFO : called evaluate()
  2022-12-29 01:32:01,527 | INFO : F-measure after evaluation: 0.8315842078960519



3. The output of ``./outputs/performance.json`` consists of dict with target metric name and its value.

.. code-block::

  {"f-measure": 0.8315842078960519}

4. ``Optional`` Additionally, we can tune testing parameters such as confidence threshold via comand line. Read more about template-specific parameters for validation in quick start [#TODO link].

For example, to increase the confidence treshold and decrease the number of False Positive predictions (there we have prediction, but don't have annotated object for it) update the evaluation comand line as it's shown below. 

Please note, that by default confidence treshold is detected automatically based on result to maximize the final F1 metric. So, to set custom confidence treshold, please disable ``result_based_confidence_threshold`` option.

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --data data.yaml 
                            --load-weights outputs/weights.pth
                            params 
                            --postprocessing.confidence_threshold 0.5
                            --postprocessing.result_based_confidence_threshold false 

  ...

  2023-01-03 18:55:01,956 | INFO : F-measure after evaluation: 0.6274238227146813

*********
Export
*********
1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format. It allows to efficiently run it on Intel hardware, especially on CPU. Also, the resulting IR model is required to run POT optimization in the section below. IR model contains of 2 files: openvino.xml for weights and openvino.bin for architecture.

2. That's how we can export the trained model ``outputs/weights.pth`` from the previous section and save the exported model to the ``outputs/openvino/`` folder.

.. code-block::

  (detection) ...$ otx export otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                              --load-weights outputs/weights.pth
                              --save-model-to outputs/openvino/

  ...

  2022-12-29 01:39:11,980 | INFO : Exporting completed
  2022-12-29 01:39:11,980 | INFO : run task done.
  2022-12-29 01:39:11,990 | INFO : Exporting completed


3. We can check the accuracy of exported model as simple as accuracy of the ``.pth`` model, using ``otx eval`` and passing IR model to ``--load-weights`` parameter.

.. code-block::

  (detection) ...$ otx eval otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml
                            --test-ann-files BCCD/valid/_annotations.coco.json 
                            --test-data-roots  BCCD/valid 
                            --load-weights outputs/openvino/openvino.xml
                            --save-performance outputs/openvino/performance.json
  
  ...
  2023-01-05 17:09:13,684 | INFO : Start OpenVINO inference
  2023-01-05 17:09:25,139 | INFO : OpenVINO inference completed
  2023-01-05 17:09:25,139 | INFO : Start OpenVINO metric evaluation
  2023-01-05 17:09:25,431 | INFO : OpenVINO metric evaluation completed
  Performance(score: 0.8315842078960519, dashboard: (1 metric groups))


*************
Optimization
*************

1. We can even more optimize the model with ``otx optimize``. It uses NNCF or POT depending on the model format.

- NNCF optimization is used for trained snapshots in a framework-specific format such as checkpoint (pth) file from Pytorch. It starts accuracy-aware quantization based on the obtained weights from the training stage. Generally, we will see the same output as during training.
- POT optimization is used for models exported in the OpenVINO™ IR format. It decreases floating-point precision to integer precision of the exported model by performing the post-training optimization.

The function results with a following files, which could be used to run :doc:`otx demo <../demo>`:

- confidence_threshold
- config.json
- label_schema.json
- openvino.bin
- openvino.xml

To learn more about optimization, refer to `NNCF repository <https://github.com/openvinotoolkit/nncf>`_.

2. Command example for optimizing a PyTorch model (.pth) with OpenVINO NNCF.

.. code-block::

  (detection) ...$ otx optimize otx/algorithms/detection/configs/detection/mobilenetv2_atss/template.yaml 
                                --train-ann-files BCCD/train/_annotations.coco.json 
                                --train-data-roots  BCCD/train 
                                --val-ann-files BCCD/valid/_annotations.coco.json 
                                --val-data-roots BCCD/valid 
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
                                --train-ann-files BCCD/train/_annotations.coco.json 
                                --train-data-roots  BCCD/train 
                                --val-ann-files BCCD/valid/_annotations.coco.json 
                                --val-data-roots BCCD/valid 
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

4. We can evaluate the optimized model passing it to ``otx eval`` function.

Now we have fully trained, optimized and exported in efficient model representation ready-to use object detection model.

Following tutorials provides further steps how to :doc:`deploy <../deploy>` and use your model in the :doc:`demonstration mode <../demo>` and visualize results.
