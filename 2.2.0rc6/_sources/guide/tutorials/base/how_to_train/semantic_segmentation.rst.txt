Semantic Segmentation model
================================

This tutorial demonstrates how to train and optimize a semantic segmentation model using the VOC2012 dataset from the PASCAL Visual Object Classes Challenge 2012.
The trained model will be used to segment images by assigning a label to each pixel of the input image.

To learn more about Segmentation task, refer to :doc:`../../../explanation/algorithms/segmentation/semantic_segmentation`.

.. note::
  To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

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

For the semnatic segmentation, we'll use the common_semantic_segmentation_dataset/supervised located at the tests/assets


*********
Training
*********

1. First of all, you need to choose which semantic segmentation model you want to train.
The list of supported recipes for semantic segmentation is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/segmentation/semantic_segmentation>`.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

          (otx) ...$ otx find --task SEMANTIC_SEGMENTATION

          ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
          ┃ Task                  ┃ Model Name                    ┃ Recipe Path                                                                        ┃
          ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
          │ SEMANTIC_SEGMENTATION │ openvino_model                │ src/otx/recipe/semantic_segmentation/openvino_model.yaml                           │
          │ SEMANTIC_SEGMENTATION │ segnext_t                     │ src/otx/recipe/semantic_segmentation/segnext_t.yaml                             │
          │ SEMANTIC_SEGMENTATION │ segnext_b                     │ src/otx/recipe/semantic_segmentation/segnext_b.yaml                        │
          │ SEMANTIC_SEGMENTATION │ dino_v2                       │ src/otx/recipe/semantic_segmentation/dino_v2.yaml                           │
          │ SEMANTIC_SEGMENTATION │ litehrnet_18                  │ src/otx/recipe/semantic_segmentation/litehrnet_18.yaml                 │
          │ SEMANTIC_SEGMENTATION │ segnext_s                     │ src/otx/recipe/semantic_segmentation/segnext_s.yaml                         │
          │ SEMANTIC_SEGMENTATION │ litehrnet_x                   │ src/otx/recipe/semantic_segmentation/litehrnet_x.yaml                         │
          │ SEMANTIC_SEGMENTATION │ litehrnet_s                   │ src/otx/recipe/semantic_segmentation/litehrnet_s.yaml                         │
          └───────────────────────┴───────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

          from otx.engine.utils.api import list_models

          model_lists = list_models(task="SEMANTIC_SEGMENTATION")
          print(model_lists)
          '''
          [
            'openvino_model',
            'dino_v2',
            'litehrnet_x',
            'litehrnet_18',
            'segnext_s',
            'segnext_b',
            'segnext_t',
            'litehrnet_s',
          ]
          '''

1.  On this step we will configure configuration
with:

- all necessary configs for litehrnet_18
- train/validation sets, based on provided annotation.

Let's prepare an OpenVINO™ Training Extensions semantic segmentation workspace running the following command:

.. code-block:: shell

  # or its config path
  (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root tests/assets/common_semantic_segmentation_dataset/supervised --print_config

  ...
  data_root: data/common_semantic_segmentation_dataset/supervised
  work_dir: otx-workspace
  callback_monitor: val/Dice
  disable_infer_num_classes: false
  engine:
    task: SEMANTIC_SEGMENTATION
    device: auto
  data:
  ...

.. note::

  If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

  .. code-block:: shell

    (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root data/common_semantic_segmentation_dataset/supervised --print_config > configs.yaml
    # Update configs.yaml & Train configs.yaml
    (otx) ...$ otx train --config configs.yaml

3. To start training we need to call ``otx train``

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/common_semantic_segmentation_dataset/supervised --task SEMANTIC_SEGMENTATION

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root data/common_semantic_segmentation_dataset/supervised

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/common_semantic_segmentation_dataset/supervised"
            recipe = "src/otx/recipe/semantic_segmentation/litehrnet_18.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/common_semantic_segmentation_dataset/supervised"

            engine = Engine(
                      model="litehrnet_18",
                      task="SEMANTIC_SEGMENTATION",
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 18 seconds with full dataset.

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
Learn more about recipe-specific parameters using ``otx train params --help``.

It can be done by manually updating parameters in the ``configs.yaml`` file in your workplace or via the command line.

For example, to decrease the batch size to 4, fix the number of epochs to 100 and disable early stopping, extend the command line above with the following line.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --data.train_subset.batch_size 4 \
                                     --max_epochs 100

    .. tab-item:: API

        .. code-block:: python

            from otx.core.config.data import SubsetConfig
            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            datamodule = OTXDataModule(..., train_subset=SubsetConfig(..., batch_size=4))

            engine = Engine(..., datamodule=datamodule)

            engine.train(max_epochs=100)


5. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
while training logs can be found in the ``{work_dir}/{timestamp}`` dir.

.. note::
  We also can visualize the training using ``Tensorboard`` as these logs are located in ``{work_dir}/{timestamp}/tensorboard``.

.. code-block::

  otx-workspace
    ├── 20240403_134256/
    |   ├── csv/
    |   ├── checkpoints/
    |   |   └── epoch_*.pth
    |   ├── tensorboard/
    |   └── configs.yaml
    └── .latest
        └── train/
  ...

After that, we have the PyTorch instance segmentation model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

***********
Validation
***********

1. ``otx test`` runs evaluation of a trained
model on a specific dataset.

The test function receives test annotation information and model snapshot, trained in the previous step.

``otx test`` will output a Dice for semantic segmentation.

2. The command below will run validation on our dataset
and save performance results in ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/Dice            │   0.1556396484375         │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/semantic_segmentation/maskrcnn_r50.yaml \
                                --data_root tests/assets/common_semantic_segmentation_dataset/supervised \
                                --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_059.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/Dice            │   0.1556396484375         │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


3. The output of ``{work_dir}/{timestamp}/csv/version_0/metrics.csv`` consists of
a dict with target metric name and its value.


*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the
OpenVINO™ Intermediate Representation (IR) format.

It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run PTQ optimization. IR model consists of 2 files: ``exported_model.xml`` for weights and ``exported_model.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``{work_dir}/{timestamp}/`` folder.

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx export --work_dir otx-workspace
            ...
            Elapsed time: 0:00:06.588245

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_033.ckpt
            ...
            Elapsed time: 0:00:06.588245

    .. tab-item:: API

        .. code-block:: python

            engine.export()


*************
Optimization
*************

1. We can further optimize the model with ``otx optimize``.
It uses NNCF or PTQ depending on the model and transforms it to ``INT8`` format.

Please, refer to :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section to get the intuition of what we use under the hood for optimization purposes.

2.  Command example for optimizing
OpenVINO™ model (.xml) with OpenVINO™ PTQ.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx optimize  --work_dir otx-workspace \
                                     --checkpoint otx-workspace/20240312_052847/exported_model.xml

            ...
            Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 30/30 • 0:00:14 • 0:00:00
            Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 58/58 • 0:00:02 • 0:00:00
            Elapsed time: 0:00:24.958733

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_052847/exported_model.xml"
            engine.optimize(checkpoint=ckpt_path)

Please note, that PTQ will take some time (generally less than NNCF optimization) without logging to optimize the model.

.. note::

    You can also pass ``export_demo_package=True`` parameter to obtain ``exportable_code.zip`` archive with packed optimized model and demo package. Please refer to :doc:`export tutorial <../export>`.

3. Finally, we can also evaluate the optimized model by passing
it to the ``otx test`` function.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \
                                --checkpoint otx-workspace/20240312_055042/optimized_model.xml \
                                --engine.device cpu

            ...
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │       test/map_50         │    0.5482384562492371     │
            └───────────────────────────┴───────────────────────────┘
            Elapsed time: 0:00:10.260521

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_055042/optimized_model.xml"
            engine.test(checkpoint=ckpt_path)

3. Now we have fully trained, optimized and exported an
efficient model representation ready-to-use semantic segmentation model.
