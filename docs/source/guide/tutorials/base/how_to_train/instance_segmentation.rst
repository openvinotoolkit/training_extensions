Instance Segmentation model
================================

This tutorial reveals end-to-end solution from installation to model export and optimization for instance segmentation task on a specific example.
On this page, we show how to train, validate, export and optimize Mask-RCNN model on a toy dataset.

To learn more about Instance Segmentation task, refer to :doc:`../../../explanation/algorithms/segmentation/instance_segmentation`.


.. note::

  To learn deeper how to manage training process of the model including additional parameters and its modification.

  To learn how to deploy the trained model, refer to: :doc:`../deploy`.

  To learn how to run the demo and visualize results, refer to: :doc:`../demo`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block:: shell

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate


***************************
Dataset preparation
***************************

..  note::

  Currently, we support the following instance segmentation dataset formats:

  - `COCO <https://cocodataset.org/#format-data>`_


1. Clone a repository with
`WGISD dataset <https://github.com/thsant/wgisd>`_.

.. code-block:: shell

  mkdir data ; cd data
  git clone https://github.com/thsant/wgisd.git
  cd wgisd
  git checkout 6910edc5ae3aae8c20062941b1641821f0c30127


This dataset contains images of grapevines with the annotation for different varieties of grapes.

- ``CDY`` - Chardonnay
- ``CFR`` - Cabernet Franc
- ``CSV`` - Cabernet Sauvignon
- ``SVB`` - Sauvignon Blanc
- ``SYH`` - Syrah

|

.. image:: ../../../../../utils/images/wgisd_dataset_sample.jpg
  :width: 600
  :alt: this image uploaded from this `source <https://github.com/thsant/wgisd/blob/master/data/CDY_2015.jpg>`_

|

2. Check the file structure of downloaded dataset,
we will need the following file structure:

.. code-block:: shell

  wgisd
  ├── annotations/
      ├── instances_train.json
      ├── instances_val.json
      (Optional)
      └── instances_test.json
  ├──images/
      (Optional)
      ├── train
      ├── val
      └── test
  (There may be more extra unrelated folders)

We can do that by running these commands:

.. code-block:: shell

  # format images folder
  mv data images

  # format annotations folder
  mv coco_annotations annotations

  # rename annotations to meet *_train.json pattern
  mv annotations/train_bbox_instances.json annotations/instances_train.json
  mv annotations/test_bbox_instances.json annotations/instances_val.json
  cp annotations/instances_val.json annotations/instances_test.json

  cd ../..

..  note::
  We can use this dataset in the detection tutorial. refer to :doc:`./detection`.

*********
Training
*********

1. First of all, you need to choose which instance segmentation model you want to train.
The list of supported templates for instance segmentation is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/segmentation/instance_segmentation>`.

  To modify the architecture of supported models with various backbones, please refer to the :doc:`advanced tutorial for backbone replacement <../../advanced/backbones>`.

.. code-block:: shell

  (otx) ...$ otx find --task INSTANCE_SEGMENTATION

  ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓               
  ┃ Task                  ┃ Model Name                    ┃ Recipe Path                                                                        ┃               
  ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩               
  │ INSTANCE_SEGMENTATION │ openvino_model                │ src/otx/recipe/instance_segmentation/openvino_model.yaml                           │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_r50                  │ src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml                             │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_r50_tile             │ src/otx/recipe/instance_segmentation/maskrcnn_r50_tile.yaml                        │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_swint                │ src/otx/recipe/instance_segmentation/maskrcnn_swint.yaml                           │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_efficientnetb2b      │ src/otx/recipe/instance_segmentation/maskrcnn_efficientnetb2b.yaml                 │               
  │ INSTANCE_SEGMENTATION │ rtmdet_inst_tiny              │ src/otx/recipe/instance_segmentation/rtmdet_inst_tiny.yaml                         │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_efficientnetb2b_tile │ src/otx/recipe/instance_segmentation/maskrcnn_efficientnetb2b_tile.yaml            │               
  │ INSTANCE_SEGMENTATION │ maskrcnn_swint_tile           │ src/otx/recipe/instance_segmentation/maskrcnn_swint_tile.yaml                      │               
  └───────────────────────┴───────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

2. On this step we will configure configuration
with:

- all necessary configs for maskrcnn_r50
- train/validation sets, based on provided annotation.

It may be counterintuitive, but for ``--data_root`` we need to pass the path to the dataset folder root (in our case it's ``data/wgisd``) instead of the folder with validation images.
This is because the function automatically detects annotations and images according to the expected folder structure we achieved above.

Let's check the object detection configuration running the following command:

.. code-block:: shell

  # or its config path
  (otx) ...$ otx train --config src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml --data_root data/wgisd --print_config

  ...
  data_root: data/wgisd
  work_dir: otx-workspace
  callback_monitor: val/map_50
  disable_infer_num_classes: false
  engine:
    task: INSTANCE_SEGMENTATION
    device: auto
  data:
  ...

.. note::

  If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

  .. code-block:: shell

    (otx) ...$ otx train --config src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml --data_root data/wgisd --print_config > configs.yaml
    # Update configs.yaml & Train configs.yaml
    (otx) ...$ otx train --config configs.yaml

3. To start training we need to call ``otx train``

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tabs::

    .. tab:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/wgisd --task INSTANCE_SEGMENTATION

    .. tab:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml --data_root data/wgisd

    .. tab:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/wgisd"
            recipe = "src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/wgisd"

            engine = Engine(
                      model="maskrcnn_r50",
                      task="INSTANCE_SEGMENTATION",
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

.. note::

  Because the dataset structure is mostly the same as detection, INSTANCE_SEGMENTATION requires the task type to be specified to enable auto-configuration.

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 20 minutes with full dataset.

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
Learn more about template-specific parameters using ``otx train params --help``.

It can be done by manually updating parameters in the ``template.yaml`` file in your workplace or via the command line.

For example, to decrease the batch size to 4, fix the number of epochs to 100 and disable early stopping, extend the command line above with the following line.

.. code-block::

                      otx train params --learning_parameters.batch_size 4 \
                              --learning_parameters.num_iters 100 \
                              --learning_parameters.enable_early_stopping false

5. The training results are ``weights.pth`` and ``label_schema.json`` files located in ``outputs/**_train/models`` folder,
while training logs can be found in the ``outputs/**_train/logs`` dir.

- ``weights.pth`` - a model snapshot
- ``label_schema.json`` - a label schema used in training, created from a dataset

These are needed as inputs for the further commands: ``export``, ``eval``,  ``optimize``,  ``deploy`` and ``demo``.

.. note::
  We also can visualize the training using ``Tensorboard`` as these logs are located in ``outputs/**/logs/**/tf_logs``.

.. code-block::

  otx-workspace-INSTANCE_SEGMENTATION
  ├── outputs/
      ├── 20230403_134256_train/
          ├── logs/
          ├── models/
              ├── weights.pth
              └── label_schema.json
          └── cli_report.log
      ├── latest_trained_model
          ├── logs/
          ├── models/
          └── cli_report.log
  ...

After that, we have the PyTorch instance segmentation model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

***********
Validation
***********

1. ``otx eval`` runs evaluation of a trained
model on a specific dataset.

The eval function receives test annotation information and model snapshot, trained in the previous step.
Please note, ``label_schema.json`` file contains meta information about the dataset and it should be located in the same folder as the model snapshot.

``otx eval`` will output a F-measure for instance segmentation.

2. The command below will run validation on our dataset
and save performance results in ``outputs/**_eval/performance.json`` file:

.. code-block::

  (otx) ...$ otx eval --test-data-roots <data_root_path>/wgisd

We will get a similar to this validation output:

.. code-block::

  ...

  2023-04-26 12:46:27,856 | INFO : Inference completed
  2023-04-26 12:46:27,856 | INFO : called evaluate()
  2023-04-26 12:46:28,453 | INFO : F-measure after evaluation: 0.5576271186440678
  2023-04-26 12:46:28,453 | INFO : Evaluation completed
  Performance(score: 0.5576271186440678, dashboard: (1 metric groups))

.. note::

  You can omit ``--test-data-roots`` if you are currently inside a workspace and have test-data stuff written in ``data.yaml``.

  Also, if you're inside a workspace and ``weights.pth`` exists in ``outputs/latest_train_model/models`` dir,
  you can omit ``--load-weights`` as well, assuming those weights are the default as ``latest_train_model/models/weights.pth``.


The output of ``./outputs/**_eval/performance.json`` consists of a dict with target metric name and its value.

.. code-block::

  {"f-measure": 0.5576271186440678}

*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the
OpenVINO™ Intermediate Representation (IR) format.

It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run PTQ optimization. IR model consists of 2 files: ``openvino.xml`` for weights and ``openvino.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``outputs/**_export/openvino`` folder.

.. note::

  if you're inside a workspace and ``weights.pth`` exists in ``outputs/latest_train_model/models`` dir,
  you can omit ``--load-weights`` as well, assuming those weights are the default as ``latest_train_model/models/weights.pth``.

.. code-block::

  (otx) ...$ otx export

  ...
  [ SUCCESS ] Generated IR version 11 model.
  [ SUCCESS ] XML file: otx-workspace-INSTANCE_SEGMENTATION/outputs/20230426_124738_export/logs/model.xml
  [ SUCCESS ] BIN file: otx-workspace-INSTANCE_SEGMENTATION/outputs/20230426_124738_export/logs/model.bin

  2023-04-26 12:47:48,293 - mmdeploy - INFO - Successfully exported OpenVINO model: outputs/20230426_124738_export/logs/model_ready.xml
  2023-04-26 12:47:48,670 | INFO : Exporting completed

*************
Optimization
*************

1. We can further optimize the model with ``otx optimize``.
It uses NNCF or PTQ depending on the model and transforms it to ``INT8`` format.

Please, refer to :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section to get the intuition of what we use under the hood for optimization purposes.

2. Command example for optimizing
a PyTorch model (`.pth`) with OpenVINO™ `NNCF <https://github.com/openvinotoolkit/nncf>`_.

.. note::

  if you're inside a workspace and ``weights.pth`` exists in ``outputs/latest_train_model/models`` dir,
  you can omit ``--load-weights`` as well (nncf only), assuming those weights are the default as ``latest_train_model/models/weights.pth``.

.. code-block::

  (otx) ...$ otx optimize

3.  Command example for optimizing
OpenVINO™ model (.xml) with OpenVINO™ PTQ.

.. code-block::

  (otx) ...$ otx optimize --load-weights openvino_model/openvino.xml

Please note, that PTQ will take some time (generally less than NNCF optimization) without logging to optimize the model.

4. Now we have fully trained, optimized and exported an
efficient model representation ready-to-use instance segmentation model.

The following tutorials provide further steps on how to :doc:`deploy <../deploy>` and use your model in the :doc:`demonstration mode <../demo>` and visualize results.
