Visual Prompting model
======================

This tutorial reveals end-to-end solution from installation to model export and optimization for visual prompting task on a specific example.
On this page, we show how to train, validate, export and optimize SegmentAnything model on a toy dataset.

To learn more about Visual Prompting task, refer to :doc:`../../../explanation/algorithms/visual_prompting/index`.

.. note::

  To learn deeper how to manage training process of the model including additional parameters and its modification.

The process has been tested on the following configuration.

- Ubuntu 18.04
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

  Currently, we support the following visual prompting dataset formats:

  - `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/common_semantic_segmentation.html>`_
  - `COCO <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/coco.html>`_
  - `Pascal VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_
  - `Datumaro <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/datumaro.html>`_

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
        (Optional)
        └── instances_test.json
    ├──images/
        (The split on folders is optional)
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

*********
Training
*********

1. First of all, you need to choose which visual prompting model you want to train.
The list of supported templates for visual prompting is available with the command line below.

.. note::

    The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/visual_prompting/index>`.


.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx find --task VISUAL_PROMPTING
            ┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Task             ┃ Model Name     ┃ Recipe Path                                              ┃
            ┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ VISUAL_PROMPTING │ sam_tiny_vit   │ src/otx/recipe/visual_prompting/sam_tiny_vit.yaml        │
            │ VISUAL_PROMPTING │ openvino_model │ src/otx/recipe/visual_prompting/openvino_model.yaml      │
            │ VISUAL_PROMPTING │ sam_vit_b      │ src/otx/recipe/visual_prompting/sam_vit_b.yaml           │
            └──────────────────┴────────────────┴──────────────────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="VISUAL_PROMPTING")
            print(model_lists)
            '''
            ['sam_tiny_vit', 'sam_vit_b', 'openvino_model']
            '''

2. On this step we will configure configuration
with:

- all necessary configs for sam_tiny_vit
- train/validation sets, based on provided annotation.

It may be counterintuitive, but for ``--data_root`` we need to pass the path to the dataset folder root (in our case it's ``data/wgisd``) instead of the folder with validation images.
This is because the function automatically detects annotations and images according to the expected folder structure we achieved above.

Let's check the visual prompting configuration running the following command:

.. code-block:: shell

    # or its config path
    (otx) ...$ otx train --config  src/otx/recipe/visual_prompting/sam_tiny_vit.yaml \
                         --data_root data/wgisd \
                         --work_dir otx-workspace \
                         --print_config

    ...
    data_root: data/wgisd
    work_dir: otx-workspace
    callback_monitor: val/f1-score
    disable_infer_num_classes: false
    engine:
      task: VISUAL_PROMPTING
      device: auto
    data:
    ...

.. note::

    If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

    .. code-block:: shell

        (otx) ...$ otx train --config  src/otx/recipe/visual_prompting/sam_tiny_vit.yaml --data_root data/wgisd --print_config > configs.yaml
        # Update configs.yaml & Train configs.yaml
        (otx) ...$ otx train --config configs.yaml

3. ``otx train`` trains a model (a particular model template)
on a dataset and results:

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$ otx train --data_root data/wgisd --task VISUAL_PROMPTING

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/visual_prompting/sam_tiny_vit.yaml --data_root data/wgisd

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/wgisd"
            recipe = "src/otx/recipe/visual_prompting/sam_tiny_vit.yaml"

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
                      model="sam_tiny_vit",
                      task="VISUAL_PROMPTING",
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

.. note::

  Because the dataset structure is mostly the same as detection, VISUAL_PROMPTING requires the task type to be specified to enable auto-configuration.

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
Learn more about template-specific parameters using ``otx train params --help``.

It can be done by manually updating parameters in the ``template.yaml`` file in your workplace or via the command line.

For example, to increase the batch size to 4, fix the number of epochs to 100, extend the command line above with the following line.

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

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 4 minutes.

After that, we have the PyTorch visual prompting model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

***********
Evaluation
***********

1. ``otx test`` runs evaluation of a
trained model on a particular dataset.

Test function receives test annotation information and model snapshot, trained in previous step.

The default metric is f1-score measure.

2. That's how we can evaluate the snapshot in ``otx-workspace``
folder on WGISD dataset and save results to ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.0009265312692150474   │
            │         test/dice         │    0.9201090335845947     │
            │       test/f1-score       │    0.9201071262359619     │
            │         test/iou          │    0.8520355224609375     │
            │      test/iter_time       │    0.3015514016151428     │
            │         test/map          │    0.5886790156364441     │
            │        test/map_50        │    0.9061686396598816     │
            │        test/map_75        │    0.6716098785400391     │
            │      test/map_large       │    0.7401198148727417     │
            │      test/map_medium      │    0.5705212950706482     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │    0.21598181128501892    │
            │        test/mar_1         │    0.03824029490351677    │
            │        test/mar_10        │    0.3468073010444641     │
            │       test/mar_100        │     0.614170253276825     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │     0.766523003578186     │
            │      test/mar_medium      │     0.599896252155304     │
            │      test/mar_small       │    0.2501521706581116     │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config src/otx/recipe/visual_prompting/sam_tiny_vit.yaml \
                                --data_root data/wgisd \
                                --checkpoint otx-workspace/.latest/train/checkpoints/epoch_009.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.0009342798730358481   │
            │         test/dice         │    0.9201090335845947     │
            │       test/f1-score       │    0.9201071262359619     │
            │         test/iou          │    0.8520355224609375     │
            │      test/iter_time       │    0.31654438376426697    │
            │         test/map          │    0.5886790156364441     │
            │        test/map_50        │    0.9061686396598816     │
            │        test/map_75        │    0.6716098785400391     │
            │      test/map_large       │    0.7401198148727417     │
            │      test/map_medium      │    0.5705212950706482     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │    0.21598181128501892    │
            │        test/mar_1         │    0.03824029490351677    │
            │        test/mar_10        │    0.3468073010444641     │
            │       test/mar_100        │     0.614170253276825     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │     0.766523003578186     │
            │      test/mar_medium      │     0.599896252155304     │
            │      test/mar_small       │    0.2501521706581116     │
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
Also, the resulting IR model is required to run PTQ optimization in the section below. IR model contains 4 files: ``exported_model_image_encoder.xml`` and ``exported_model_decoder.xml`` for architecture and ``exported_model_image_encoder.bin`` and ``exported_model_decoder.bin`` for weights.

2. That's how we can export the trained model ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt``
from the previous section and save the exported model to the ``{work_dir}/{timestamp}/`` folder.

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx export --work_dir otx-workspace
            ...
            Elapsed time: 0:00:05.396129

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/.latest/train/checkpoints/epoch_009.ckpt
            ...
            Elapsed time: 0:00:05.313879

    .. tab-item:: API

        .. code-block:: python

            engine.export()


3. We can check the accuracy of the IR model and the consistency between the exported model and the PyTorch model,
using ``otx test`` and passing the IR model path to the ``--checkpoint`` parameter.

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \
                                --checkpoint otx-workspace/.latest/export/exported_model_decoder.xml \
                                --engine.device cpu
            ...
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.0007283412269316614   │
            │         test/dice         │    0.9990837574005127     │
            │       test/f1-score       │    0.9169966578483582     │
            │         test/iou          │    0.8467163443565369     │
            │      test/iter_time       │    3.1121630668640137     │
            │         test/map          │    0.47309553623199463    │
            │        test/map_50        │    0.8371172547340393     │
            │        test/map_75        │    0.5044668912887573     │
            │      test/map_large       │    0.6876431107521057     │
            │      test/map_medium      │    0.5046071410179138     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │    0.11672457307577133    │
            │        test/mar_1         │    0.02601064182817936    │
            │        test/mar_10        │    0.26142847537994385    │
            │       test/mar_100        │    0.6027402281761169     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │    0.7594292163848877     │
            │      test/mar_medium      │    0.5897444486618042     │
            │      test/mar_small       │    0.22268414497375488    │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config src/otx/recipe/visual_prompting/sam_tiny_vit.yaml \
                                --data_root data/wgisd \
                                --checkpoint otx-workspace/.latest/export/exported_model_decoder.xml \
                                --engine.device cpu
            ...
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.0007516053738072515   │
            │         test/dice         │    0.9990837574005127     │
            │       test/f1-score       │    0.9169966578483582     │
            │         test/iou          │    0.8467163443565369     │
            │      test/iter_time       │     3.09753680229187      │
            │         test/map          │    0.47309553623199463    │
            │        test/map_50        │    0.8371172547340393     │
            │        test/map_75        │    0.5044668912887573     │
            │      test/map_large       │    0.6876431107521057     │
            │      test/map_medium      │    0.5046071410179138     │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │    0.11672457307577133    │
            │        test/mar_1         │    0.02601064182817936    │
            │        test/mar_10        │    0.26142847537994385    │
            │       test/mar_100        │    0.6027402281761169     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │    0.7594292163848877     │
            │      test/mar_medium      │    0.5897444486618042     │
            │      test/mar_small       │    0.22268414497375488    │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            exported_model = engine.export()
            engine.test(checkpoint=exported_model)

.. note::
    Visual prompting task has two IR models unlike other tasks. 
    But it doesn't matter which one to be inserted to ``--checkpoint`` for testing because OTX will automatically load both IR models.


4. ``Optional`` Additionally, we can tune confidence threshold via the command line.
Learn more about template-specific parameters using ``otx export --help``.

For example, If you want to get the ONNX model format you can run it like below.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/.latest/train/checkpoints/epoch_009.ckpt --export_format ONNX

    .. tab-item:: API

        .. code-block:: python

            engine.export(..., export_format="ONNX")


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
                                     --checkpoint otx-workspace/.latest/export/exported_model_decoder.xml \
                                     --data.config.train_subset.num_workers 0

            ...
            Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 55/55 • 0:00:35 • 0:00:00
            Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 50/50 • 0:00:01 • 0:00:00
            Elapsed time: 0:04:28.609954

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/.latest/export/exported_model_decoder.xml"
            engine.optimize(checkpoint=ckpt_path)

The optimization time highly relies on the hardware characteristics, for example on Intel(R) Core(TM) i9-11900 it took about 10 minutes.
Please note, that PTQ will take some time without logging to optimize the model.

.. note::
    Optimization is performed in the following order: image encoder  then decoder. 
    Because the optimized image encoder is used for decoder optimization, segmentation fault error can occur when releasing threads and memories after optimization step.
    It doesn't affect optimization results, but it's recommended to set ``--data.config.train_subset.num_workers 0`` to avoid this error.

3. Finally, we can also evaluate the optimized model by passing
it to the ``otx test`` function.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \ 
                                --checkpoint otx-workspace/.latest/optimize/optimized_model_decoder.xml \
                                --engine.device cpu

            ...
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │   0.0007440951885655522   │
            │         test/dice         │     0.998199462890625     │
            │       test/f1-score       │     0.853766918182373     │
            │         test/iou          │    0.7448458075523376     │
            │      test/iter_time       │    2.8865625858306885     │
            │         test/map          │    0.23295165598392487    │
            │        test/map_50        │    0.5494663119316101     │
            │        test/map_75        │    0.15102604031562805    │
            │      test/map_large       │    0.45290130376815796    │
            │      test/map_medium      │    0.16153287887573242    │
            │    test/map_per_class     │           -1.0            │
            │      test/map_small       │   0.012729672715067863    │
            │        test/mar_1         │   0.014449129812419415    │
            │        test/mar_10        │    0.15996699035167694    │
            │       test/mar_100        │    0.3901452422142029     │
            │  test/mar_100_per_class   │           -1.0            │
            │      test/mar_large       │    0.5868775844573975     │
            │      test/mar_medium      │    0.30925533175468445    │
            │      test/mar_small       │   0.027198636904358864    │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/.latest/optimize/optimized_model_decoder.xml"
            engine.test(checkpoint=ckpt_path)

.. note::
    Visual prompting task has two IR models unlike other tasks. 
    But it doesn't matter which one to be inserted to ``--checkpoint`` for testing because OTX will automatically load both IR models.

Now we have fully trained, optimized and exported an efficient model representation ready-to-use visual prompting model.
