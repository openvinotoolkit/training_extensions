Action Detection  model
================================

This live example shows how to easily train and validate for spatio-temporal action detection model on the subset of `JHMDB <http://jhmdb.is.tue.mpg.de/>`_. 
To learn more about Action Detection task, refer to :doc:`../../../explanation/algorithms/action/action_detection`. 

.. note::

  To learn deeper how to manage training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.1

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/quick_start_guide/installation>` 
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual 
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate


***************************
Dataset preparation
***************************

Although we offer conversion codes from `ava dataset format <https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ava/README.md>`_ to `cvat dataset format <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_ from `this code <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/action/utils/convert_public_data_to_cvat.py>`_, for easy beginning you can download subset of JHMDB dataset, which already transformed to CVAT format from `this link <https://drive.google.com/file/d/1ZgUYkhOAJ9_-xMCujPJlMLFILuGkhI4X/view?usp=share_link>`_.

If you download data from link and extract to ``training_extensions/data`` folder(you should make data folder at first), you can see the structure below:

.. code-block::

    training_extensions
    └── data
        └── JHMDB_5%
            ├── train
            │    └── brush_hair_Brushing_Hair_with_Beth_brush_hair_h_nm_np1_le_goo_0
            │        ├── annotations.xml
            │        └── images [40 frames]
            │
            │── test
            │    └── brush_hair_Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_fr_med_0
            │        ├── annotations.xml
            │        └── images [40 frames]
            │
            │── train.pkl
            └── test.pkl


*********
Training
*********

1. First of all, we need to choose which action detection model we will train.
The list of supported templates for action detection is available with the command line below:

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/action/action_detection>`.

.. code-block::

  (otx) ...$ otx find --task action_detection

  +------------------+---------------------------------------+---------------+---------------------------------------------------------------------+
  |       TASK       |                   ID                  |      NAME     |                              BASE PATH                              |
  +------------------+---------------------------------------+---------------+---------------------------------------------------------------------+
  | ACTION_DETECTION | Custom_Action_Detection_X3D_FAST_RCNN | X3D_FAST_RCNN | otx/algorithms/action/configs/detection/x3d_fast_rcnn/template.yaml |
  +------------------+---------------------------------------+---------------+---------------------------------------------------------------------+

To have a specific example in this tutorial, all commands will be run on the X3D_FAST_RCNN  model. It's a light model, that achieves competitive accuracy while keeping the inference fast.

2. Next, we need to create workspace
for various tasks we provide.

Let's prepare an OpenVINO™ Training Extensions action detection workspace running the following command:

.. code-block::

  (otx) ...$ otx build --train-data-roots ./data/JHMDB_5%/train --val-data-roots ./data/JHMDB_5%/test --model X3D_FAST_RCNN

  [*] Workspace Path: otx-workspace-ACTION_DETECTION
  [*] Load Model Template ID: Custom_Action_Detection_X3D_FAST_RCNN
  [*] Load Model Name: X3D_FAST_RCNN
  [*]     - Updated: otx-workspace-ACTION_DETECTION/model.py
  [*]     - Updated: otx-workspace-ACTION_DETECTION/data_pipeline.py
  [*] Update data configuration file to: otx-workspace-ACTION_DETECTION/data.yaml

  (otx) ...$ cd ./otx-workspace-ACTION_DETECTION

It will create **otx-workspace-ACTION_DETECTION** with all necessary configs for X3D_FAST_RCNN, prepared ``data.yaml`` to simplify CLI commands launch and splitted dataset.

3. To start training we need to call ``otx train``
command in our workspace:

.. code-block::

  (otx) ...$ otx train

That's it! The training will return artifacts: ``weights.pth`` and ``label_schema.json``, which are needed as input for the further commands: ``export``, ``eval``,  ``optimize``,  etc.

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 70 minutes.

After that, we have the PyTorch action detection model trained with OpenVINO™ Training Extensions.

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained
model on a specific dataset.

The eval function receives test annotation information and model snapshot, trained in the previous step.
Please note, ``label_schema.json`` file contains meta information about the dataset and it should be located in the same folder as the model snapshot.

``otx eval`` will output a mAP score for spatio-temporal action detection.

2. The command below will run validation on our dataset
and save performance results in ``performance.json`` file:

.. code-block::

  (otx) ...$ otx eval --test-data-roots ../data/JHMDB_5%/test \
                      --load-weights models/weights.pth \
                      --save-performance performance.json

We will get a similar to this validation output after some validation time (about 2 minutes):

.. code-block::

  2023-02-21 22:42:14,540 - mmaction - INFO - Loaded model weights from Task Environment
  2023-02-21 22:42:14,540 - mmaction - INFO - Model architecture: X3D_FAST_RCNN
  2023-02-21 22:42:14,739 - mmaction - INFO - Patching pre proposals...
  2023-02-21 22:42:14,749 - mmaction - INFO - Done.
  2023-02-21 22:44:24,345 - mmaction - INFO - Inference completed
  2023-02-21 22:44:24,347 - mmaction - INFO - called evaluate()
  2023-02-21 22:44:26,349 - mmaction - INFO - Final model performance: Performance(score: 0.537625754527163, dashboard: (1 metric groups))
  2023-02-21 22:44:26,349 - mmaction - INFO - Evaluation completed
  Performance(score: 0.537625754527163, dashboard: (1 metric groups))

.. note::

  Currently we don't support export and optimize task in action detection. We will support these features very near future.
