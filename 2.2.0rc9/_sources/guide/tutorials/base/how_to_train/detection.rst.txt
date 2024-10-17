Object Detection model
======================

This tutorial reveals end-to-end solution from installation to model export and optimization for object detection task on a specific example.

To learn more about Object Detection task, refer to :doc:`../../../explanation/algorithms/object_detection/object_detection`.

On this page, we show how to train, validate, export and optimize ATSS model on WGISD public dataset.

To have a specific example in this tutorial, all commands will be run on the ATSS model. It's a medium model, that achieves relatively high accuracy while keeping the inference fast.

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


.. _wgisd_dataset_descpiption:

***************************
Dataset preparation
***************************

..  note::

    Currently, we support the following object detection dataset formats:

    - `COCO <https://cocodataset.org/#format-data>`_
    - `Pascal-VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_
    - `YOLO <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/yolo.html>`_

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

It's a great example to start with. The model achieves high accuracy right from the beginning of the training due to relatively large and focused objects. Also, these objects are distinguished by a person, so we can check inference results just by looking at images.

|

.. image:: ../../../../../utils/images/wgisd_gt_sample.jpg
  :width: 600
  :alt: this image uploaded from this `source <https://github.com/thsant/wgisd/blob/master/data/CDY_2015.jpg>`_

|

2. To run the training using :doc:`auto-configuration feature <../../../explanation/additional_features/auto_configuration>`,
we need to reformat the dataset according to this structure:

.. code-block:: shell

    wgisd
    ├── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
    ├──images/
        ├── train
        ├── val
        └── test

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

*********
Training
*********

1. First of all, you need to choose which object detection model you want to train.
The list of supported recipes for object detection is available with the command line below.

.. note::

    The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/object_detection/object_detection>`.


.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx find --task DETECTION --pattern atss
            ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Task      ┃ Model Name            ┃ Recipe Path                                                    ┃
            ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ DETECTION │ atss_mobilenetv2_tile │ src/otx/recipe/detection/atss_mobilenetv2_tile.yaml            │
            │ DETECTION │ atss_resnext101       │ src/otx/recipe/detection/atss_resnext101.yaml                  │
            │ DETECTION │ atss_mobilenetv2      │ src/otx/recipe/detection/atss_mobilenetv2.yaml                 │
            └───────────┴───────────────────────┴────────────────────────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="DETECTION", pattern="atss")
            print(model_lists)
            '''
            [
                'atss_mobilenetv2',
                'atss_mobilenetv2_tile',
                'atss_resnext101',
            ]
            '''

.. _detection_workspace:

2. On this step we will configure configuration
with:

- all necessary configs for atss_mobilenetv2
- train/validation sets, based on provided annotation.

It may be counterintuitive, but for ``--data_root`` we need to pass the path to the dataset folder root (in our case it's ``data/wgisd``) instead of the folder with validation images.
This is because the function automatically detects annotations and images according to the expected folder structure we achieved above.

Let's check the object detection configuration running the following command:

.. code-block:: shell

    # or its config path
    (otx) ...$ otx train --config  src/otx/recipe/detection/atss_mobilenetv2.yaml \
                         --data_root data/wgisd \
                         --work_dir otx-workspace \
                         --print_config

    ...
    data_root: data/wgisd
    work_dir: otx-workspace
    callback_monitor: val/map_50
    disable_infer_num_classes: false
    engine:
      task: DETECTION
      device: auto
    data:
    ...

.. note::

    If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

    .. code-block:: shell

        (otx) ...$ otx train --config  src/otx/recipe/detection/atss_mobilenetv2.yaml --data_root data/wgisd --print_config > configs.yaml
        # Update configs.yaml & Train configs.yaml
        (otx) ...$ otx train --config configs.yaml


3. ``otx train`` trains a model (a particular model recipe)
on a dataset and results:

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/wgisd

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/detection/atss_mobilenetv2.yaml --data_root data/wgisd

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/wgisd"
            recipe = "src/otx/recipe/detection/atss_mobilenetv2.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/wgisd"

            engine = Engine(
                      model="atss_mobilenetv2",
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)


4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
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


5. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
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
folder on WGISD dataset and save results to ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.025369757786393166    │
            │       test/map_50         │    0.8693901896476746     │
            │      test/iter_time       │    0.08180806040763855    │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/detection/atss_mobilenetv2.yaml \
                                --data_root data/wgisd \
                                --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_033.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.025369757786393166    │
            │       test/map_50         │    0.8693901896476746     │
            │      test/iter_time       │    0.08180806040763855    │
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
Also, the resulting IR model is required to run PTQ optimization in the section below. IR model contains 2 files: ``exported_model.xml`` for architecture and ``exported_model.bin`` for weights.

2. That's how we can export the trained model ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt``
from the previous section and save the exported model to the ``{work_dir}/{timestamp}/`` folder.

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
            │         test/map          │    0.5444773435592651     │
            │        test/map_50        │    0.8693901896476746     │
            │        test/map_75        │    0.5761404037475586     │
            │      test/map_large       │     0.561242401599884     │
            │      test/map_medium      │    0.2926788330078125     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │           -1.0            │
            │        test/mar_1         │   0.055956535041332245    │
            │        test/mar_10        │    0.45759353041648865    │
            │       test/mar_100        │    0.6809769868850708     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │    0.6932432055473328     │
            │      test/mar_medium      │    0.46584922075271606    │
            │      test/mar_small       │           -1.0            │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config src/otx/recipe/detection/atss_mobilenetv2.yaml \
                                --data_root data/wgisd \
                                --checkpoint otx-workspace/20240312_052847/exported_model.xml \
                                --engine.device cpu
            ...
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │         test/map          │    0.5444773435592651     │
            │        test/map_50        │    0.8693901896476746     │
            │        test/map_75        │    0.5761404037475586     │
            │      test/map_large       │     0.561242401599884     │
            │      test/map_medium      │    0.2926788330078125     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │           -1.0            │
            │        test/mar_1         │   0.055956535041332245    │
            │        test/mar_10        │    0.45759353041648865    │
            │       test/mar_100        │    0.6809769868850708     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │    0.6932432055473328     │
            │      test/mar_medium      │    0.46584922075271606    │
            │      test/mar_small       │           -1.0            │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            exported_model = engine.export()
            engine.test(checkpoint=exported_model)


4. ``Optional`` Additionally, we can tune confidence threshold via the command line.
Learn more about recipe-specific parameters using ``otx export --help``.

For example, If you want to get the ONNX model format you can run it like below.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_033.ckpt --export_format ONNX

    .. tab-item:: API

        .. code-block:: python

            engine.export(..., export_format="ONNX")

If you also want to export ``saliency_map``, a feature related to explain, and ``feature_vector`` information for XAI, you can do the following.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_033.ckpt --explain True

    .. tab-item:: API

        .. code-block:: python

            engine.export(..., explain=True)


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
            Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 30/30 • 0:00:14 • 0:00:00
            Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 58/58 • 0:00:02 • 0:00:00
            Elapsed time: 0:00:24.958733

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_052847/exported_model.xml"
            engine.optimize(checkpoint=ckpt_path)


The optimization time highly relies on the hardware characteristics, for example on Intel(R) Core(TM) i9-11900 it took about 25 seconds.
Please note, that PTQ will take some time without logging to optimize the model.

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
            │       test/map_50         │    0.8693901896476746     │
            └───────────────────────────┴───────────────────────────┘
            Elapsed time: 0:00:10.260521

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240312_055042/optimized_model.xml"
            engine.test(checkpoint=ckpt_path)

Now we have fully trained, optimized and exported an efficient model representation ready-to-use object detection model.
