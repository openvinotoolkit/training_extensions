Action Classification model
================================

This live example shows how to easily train, validate, optimize and export classification model on the `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_.
To learn more about Action Classification task, refer to :doc:`../../../explanation/algorithms/action/action_classification`.

.. note::
  To learn more about managing the training process of the model including additional parameters and modification, refer to :doc:`./detection`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.6

.. note::

  To learn more about the model, algorithm and dataset format, refer to :doc:`action classification explanation <../../../explanation/algorithms/action/action_classification>`.


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

HMDB51 dataset can be downloaded via `mmaction guide <https://github.com/open-mmlab/mmaction2/tree/main/tools/data/hmdb51>`_
After downloading raw annotations and videos, you need to covert to `Kinetics <https://github.com/cvdfoundation/kinetics-dataset>`_ dataset form
The expected data structure is

.. code-block::

    training_extensions
    ├── data
        ├── hmdb51
        ├── test
        │   ├── Silky_Straight_Hair_Original_brush_hair_h_nm_np1_ba_goo_0.avi 
        ├── test.csv
        ├── train
        │   ├── red_head_brush_hair_u_cm_np1_ba_goo_0.avi 
        ├── train.csv
        ├── val   
        │   ├── sarah_brushing_her_hair_brush_hair_h_cm_np1_ri_goo_0.avi   
        └── val.csv

And, csv file should be 

You can check a example of Kinetics dataset format in `sample dataset <../../../../../../tests/assets/action_classification_dataset>`_

.. code-block::

    label,youtube_id,time_start,time_end,split,is_cc
    brush_hair,red_head_brush_hair_u_cm_np1_ba_goo_0,0,276,train,0


*********
Training
*********

1. You need to choose, which action classification model you want to train.
To see the list of supported recipes, run the following command:

.. note::

  OpenVINO™ Training Extensions supports X3D and MoViNet recipe now, other architecture will be supported in future.

.. code-block::

  (otx) ...$ otx find --task ACTION_CLASSIFICATION

  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+
  |          TASK         |                  Model Name          |                                         Recipe PATH                             |
  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+
  | ACTION_CLASSIFICATION | openvino_model                       | ../otx/recipe/action/action_classification/openvino_model.yaml                  |
  | ACTION_CLASSIFICATION | x3d                                  | ../otx/recipe/action/action_classification/x3d.yaml                             |
  | ACTION_CLASSIFICATION | movinet                              | ../otx/recipe/action/action_classification/movinet.yaml                         |
  +-----------------------+--------------------------------------+---------------------------------------------------------------------------------+

All commands will be run on the X3D model. It's a light model, that achieves competitive accuracy while keeping the inference fast.

2. ``otx train`` trains a model (a particular model template)
on a dataset and results:

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/hmdb51

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/action/action_classification/x3d.yaml --data_root data/hmdb51

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/hmdb51"
            recipe = "src/otx/recipe/action/action_classification/x3d.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/hmdb51"

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

After that, we have the PyTorch action classification model trained with OpenVINO™ Training Extensions, which we can use for evaliation, export, optimization and deployment.

***********
Evaluation
***********

1. ``otx test`` runs evaluation of a
trained model on a particular dataset.

Test function receives test annotation information and model snapshot, trained in previous step.

The default metric is accuracy measure.

2. That's how we can evaluate the snapshot in ``otx-workspace``
folder on hmdb51 dataset and save results to ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃        Test metric        ┃       DataLoader 0        ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
                │       test/accuracy       │    0.6039215922355652     │
                │      test/data_time       │    0.13730056583881378    │
                │      test/iter_time       │    0.16275013983249664    │
                └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/action/action_classification/x3d.yaml \
                                --data_root data/hmdb51 \
                                --checkpoint otx-workspace/20240312_051135/last.ckpt
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃        Test metric        ┃       DataLoader 0        ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
                │       test/accuracy       │    0.6039215922355652     │
                │      test/data_time       │    0.13730056583881378    │
                │      test/iter_time       │    0.16275013983249664    │
                └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


3. The output of ``{work_dir}/{timestamp}/csv/version_0/metrics.csv`` consists of
a dict with target metric name and its value.


*********
Export
*********

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows to efficiently run it on Intel hardware, especially on CPU, using OpenVINO™ runtime.
Also, the resulting IR model is required to run PTQ optimization in the section below. IR model contains 2 files: ``exported_model.xml`` for weights and ``exported_model.bin`` for architecture.

2. That's how we can export the trained model ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt``
from the previous section and save the exported model to the ``{work_dir}/{timestamp}/`` folder.

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx export --work_dir otx-workspace
            ...
                Elapsed time: 0:00:21.295829

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/20240312_051135/last.ckpt
            ...
                Elapsed time: 0:00:21.295829

    .. tab-item:: API

        .. code-block:: python

            engine.export()


3. We can check the accuracy of the IR model and the consistency between the exported model and the PyTorch model,
using ``otx test`` and passing the IR model path to the ``--checkpoint`` parameter.

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \
                                --checkpoint otx-workspace/20240312_052847/exported_model.xml \
                                --engine.device cpu
            ...
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
              ┃        Test metric        ┃       DataLoader 0        ┃
              ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
              │       test/accuracy       │    0.5222222208976746     │
              │      test/data_time       │    0.14048805832862854    │
              │      test/iter_time       │    0.5871070623397827     │
              └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config src/otx/recipe/action/action_classification/x3d.yaml \
                                --data_root data/hmdb51 \
                                --checkpoint otx-workspace/20240312_052847/exported_model.xml \
                                --engine.device cpu
            ...
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
              ┃        Test metric        ┃       DataLoader 0        ┃
              ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
              │       test/accuracy       │    0.5222222208976746     │
              │      test/data_time       │    0.14048805832862854    │
              │      test/iter_time       │    0.5871070623397827     │
              └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            exported_model = engine.export()
            engine.test(checkpoint=exported_model)


*************
Optimization
*************

1. We can further optimize the model with ``otx optimize``.
It uses PTQ depending on the model and transforms it to ``INT8`` format.

``PTQ`` optimization is used for models exported in the OpenVINO™ IR format. It decreases the floating-point precision to integer precision of the exported model by performing the post-training optimization.

To learn more about optimization, refer to `NNCF repository <https://github.com/openvinotoolkit/nncf>`_.

2.  Command example for optimizing OpenVINO™ model (.xml)
with OpenVINO™ PTQ.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx optimize  --work_dir otx-workspace \ 
                                     --checkpoint otx-workspace/20240312_052847/exported_model.xml

            ...
            Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 209/209 • 0:03:42 • 0:00:00
            Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 88/88 • 0:00:03 • 0:00:00
            Elapsed time: 0:03:51.613333

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_052847/exported_model.xml"
            engine.optimize(checkpoint=ckpt_path)


The optimization time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 it took about 10 minutes.
Please note, that PTQ will take some time without logging to optimize the model.

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
              │       test/accuracy       │    0.5078431367874146     │
              │      test/data_time       │    0.23449821770191193    │
              │      test/iter_time       │    0.4908757507801056     │
              └───────────────────────────┴───────────────────────────┘
            Elapsed time: 0:01:40.255130

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_055042/optimized_model.xml"
            engine.test(checkpoint=ckpt_path)

Now we have fully trained, optimized and exported an efficient model representation ready-to-use action_classification model