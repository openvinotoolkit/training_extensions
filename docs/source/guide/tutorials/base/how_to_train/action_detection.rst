Action Detection model
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

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
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

For action detection task, you need to prepare dataset whose format is `AVA <https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava/README.md>`_ dataset. 
For easy beginning, we provide `sample dataset <https://drive.google.com/file/d/1758dyPeFv4wS0gqL42sSXZSHWysL0Xr8/view?usp=drive_link>`_

If you download data from link and extract to ``training_extensions/data`` folder(you should make data folder at first), you can see the structure below:

.. code-block::

    training_extensions
    └── data
        └── JHMDB_10%
            ├── annotations
            │    └── ava_action_list_v2.2.pbtxt
            │    └── ava_test.csv
            │    └── ava_train.csv
            │    └── ava_val.csv
            │    └── test.pkl
            │    └── train.pkl
            │    └── val.pkl
            │
            └── frames
                │── train_video001
                │   └── train_video001_0001.jpg
                └── test_video001
                    └── test_video001_0001.jpg



*********
Training
*********

1. First of all, you need to choose which action detection model you want to train.
The list of supported recipes for action detection is available with the command line below:

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/action/action_detection>`.

.. code-block::

  (otx) ...$ otx find --task ACTION_DETECTION

  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+
  |          TASK         |                  Model Name          |                                         Recipe PATH                             |
  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+
  | ACTION_DETECTION      | x3d_fast_rcnn                        | ../otx/recipe/action/action_detection/x3d_fast_rcnn.yaml                        |
  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+

To have a specific example in this tutorial, all commands will be run on the X3D_FAST_RCNN  model. It's a light model, that achieves competitive accuracy while keeping the inference fast.

2. ``otx train`` trains a model (a particular model template)
on a dataset and results:

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/JHMDB_10%

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/action/action_detection/x3d_fast_rcnn.yaml --data_root data/JHMDB_10%

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/JHMDB_10%"
            recipe = "src/otx/recipe/action/action_detection/x3d_fast_rcnn.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/JHMDB_10%"

            engine = Engine(
                      model="x3d",
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)


3. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
Learn more about specific parameters using ``otx train --help -v`` or ``otx train --help -vv``.

For example, to decrease the batch size to 4, fix the number of epochs to 100, extend the command line above with the following line.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --data.config.train_subset.batch_size 4 \
                                     --max_epochs 100

    .. tab-item:: API

        .. code-block:: python

            from otx.core.config.data import DataModuleConfig, SubsetConfig
            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            data_config = DataModuleConfig(..., train_subset=SubsetConfig(..., batch_size=4))
            datamodule = OTXDataModule(..., config=data_config)

            engine = Engine(..., datamodule=datamodule)

            engine.train(max_epochs=100)


4. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
while training logs can be found in the ``{work_dir}/{timestamp}`` dir.

.. note::
    We also can visualize the training using ``Tensorboard`` as these logs are located in ``{work_dir}/{timestamp}/tensorboard``.

.. code-block::

    otx-workspace
    ├── 20240403_134256/
        ├── csv/
        ├── checkpoints/
        |   └── epoch_*.pth
        ├── tensorboard/
        └── configs.yaml
    └── .latest
        └── train/
    ...

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 3 minutes.

After that, we have the PyTorch object detection model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

***********
Evaluation
***********

1. ``otx test`` runs evaluation of a
trained model on a particular dataset.

Test function receives test annotation information and model snapshot, trained in previous step.

The default metric is mAP_50 measure.

2. That's how we can evaluate the snapshot in ``otx-workspace``
folder on JHMDB_10% dataset and save results to ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
              ┃        Test metric        ┃       DataLoader 0        ┃
              ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
              │      test/data_time       │   0.006367621477693319    │
              │      test/iter_time       │    0.02698644995689392    │
              │         test/map          │    0.10247182101011276    │
              │        test/map_50        │    0.3779516816139221     │
              │        test/map_75        │    0.03639398142695427    │
              │      test/map_large       │    0.11831618845462799    │
              │      test/map_medium      │    0.02958027645945549    │
              │    test/map_per_class     │           -1.0            │
              │      test/map_small       │            0.0            │
              │        test/mar_1         │    0.12753313779830933    │
              │        test/mar_10        │    0.1305265873670578     │
              │       test/mar_100        │    0.1305265873670578     │
              │  test/mar_100_per_class   │           -1.0            │
              │      test/mar_large       │    0.14978596568107605    │
              │      test/mar_medium      │    0.06217033043503761    │
              │      test/mar_small       │            0.0            │
              └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/action/action_detection/x3d_fast_rcnn.yaml \
                                --data_root data/JHMDB_10% \
                                --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_033.ckpt
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
              ┃        Test metric        ┃       DataLoader 0        ┃
              ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
              │      test/data_time       │   0.006367621477693319    │
              │      test/iter_time       │    0.02698644995689392    │
              │         test/map          │    0.10247182101011276    │
              │        test/map_50        │    0.3779516816139221     │
              │        test/map_75        │    0.03639398142695427    │
              │      test/map_large       │    0.11831618845462799    │
              │      test/map_medium      │    0.02958027645945549    │
              │    test/map_per_class     │           -1.0            │
              │      test/map_small       │            0.0            │
              │        test/mar_1         │    0.12753313779830933    │
              │        test/mar_10        │    0.1305265873670578     │
              │       test/mar_100        │    0.1305265873670578     │
              │  test/mar_100_per_class   │           -1.0            │
              │      test/mar_large       │    0.14978596568107605    │
              │      test/mar_medium      │    0.06217033043503761    │
              │      test/mar_small       │            0.0            │
              └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


3. The output of ``{work_dir}/{timestamp}/csv/version_0/metrics.csv`` consists of
a dict with target metric name and its value.