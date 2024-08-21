Anomaly Detection Tutorial
================================

This tutorial demonstrates how to train, evaluate, and deploy a classification, detection, or segmentation model for anomaly detection in industrial or medical applications.
Read :doc:`../../../explanation/algorithms/anomaly/index` for more information about the Anomaly tasks.

.. note::
    To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

The process has been tested with the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8


*****************************
Setup the Virtual environment
*****************************

1. To create a universal virtual environment for OpenVINO™ Training Extensions,
please follow the installation process in the :doc:`quick start guide <../../../get_started/installation>`.

2. Activate your virtual
environment:

.. code-block:: shell

    .otx/bin/activate
    # or by this line, if you created an environment, using tox
    . venv/otx/bin/activate

**************************
Dataset Preparation
**************************

1. For this example, we will use the `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ dataset.
You can download the dataset from the link above. We will use the ``bottle`` category for this tutorial.

2. This is how it might look like in your
file system:

.. code-block::

    datasets/MVTec/bottle
    ├── ground_truth
    │   ├── broken_large
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ├── 002_mask.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ...
    │   └── contamination
    │       ├── 000_mask.png
    │       ├── 001_mask.png
    │       ...
    ├── license.txt
    ├── readme.txt
    ├── test
    │   ├── broken_large
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── contamination
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   └── good
    │       ├── 000.png
    │       ├── 001.png
    │       ...
    └── train
        └── good
            ├── 000.png
            ├── 001.png
            ...

***************************
Training
***************************

1. For this example let's look at the
anomaly detection tasks

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$  otx find --task ANOMALY_DETECTION
            ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                 
            ┃ Task              ┃ Model Name ┃ Recipe Path                                 ┃                                 
            ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩                                 
            │ ANOMALY_DETECTION │ stfpm      │ src/otx/recipe/anomaly_detection/stfpm.yaml │                                 
            │ ANOMALY_DETECTION │ padim      │ src/otx/recipe/anomaly_detection/padim.yaml │                                 
            └───────────────────┴────────────┴─────────────────────────────────────────────┘ 

    .. tab-item:: API

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="ANOMALY_DETECTION")
            print(model_lists)
            '''
            ['stfpm', 'padim']
            '''

You can see two anomaly detection models, STFPM and PADIM. For more detail on each model, refer to Anomalib's `STFPM <https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/models/image/stfpm.html>`_ and `PADIM <https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/models/image/padim.html>`_ documentation.

2. Let's proceed with PADIM for
this example.

.. tab-set::

    .. tab-item:: CLI (auto-config)

        .. code-block:: shell

            (otx) ...$  otx train --data_root datasets/MVTec/bottle \
                                  --task ANOMALY_DETECTION

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$  otx train --config src/otx/recipe/anomaly_detection/padim.yaml \
                                  --data_root datasets/MVTec/bottle

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "datasets/MVTec/bottle"
            recipe = "src/otx/recipe/anomaly_detection/padim.yaml"

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            data_root = "datasets/MVTec/bottle"

            engine = Engine(
                        model="padim",
                        data_root=data_root,
                        task="ANOMALY_DETECTION",
                        work_dir="otx-workspace",
                    )

            engine.train(...)


3. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs.
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

This will start training and generate artifacts for commands such as ``export`` and ``optimize``. You will notice the ``otx-workspace`` directory in your current working directory. This is where all the artifacts are stored.

**************
Evaluation
**************

Now we have trained the model, let's see how it performs on a specific dataset. In this example, we will use the same dataset to generate evaluation metrics. To perform evaluation you need to run the following commands:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │        image_AUROC        │            0.8            │
            │       image_F1Score       │            0.8            │
            │        pixel_AUROC        │            0.8            │
            │       pixel_F1Score       │            0.8            │
            │      test/data_time       │    0.6517705321311951     │
            │      test/iter_time       │    0.6630784869194031     │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/anomaly_detection/padim.yaml \
                                --data_root datasets/MVTec/bottle \
                                --checkpoint otx-workspace/20240313_042421/checkpoints/epoch_010.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │        image_AUROC        │            0.8            │
            │       image_F1Score       │            0.8            │
            │        pixel_AUROC        │            0.8            │
            │       pixel_F1Score       │            0.8            │
            │      test/data_time       │    0.6517705321311951     │
            │      test/iter_time       │    0.6630784869194031     │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


The primary metric here is the f-measure computed against the ground-truth bounding boxes. It is also called the local score. In addition, f-measure is also used to compute the global score. The global score is computed based on the global label of the image. That is, the image is anomalous if it contains at least one anomaly. This global score is stored as an additional metric.

.. note::

    All task types report Image-level F-measure as the primary metric. In addition, both localization tasks (anomaly detection and anomaly segmentation) also report localization performance (F-measure for anomaly detection and Dice-coefficient for anomaly segmentation).

*******
Export
*******

1. ``otx export`` exports a trained Pytorch `.pth` model to the OpenVINO™ Intermediate Representation (IR) format.
It allows running the model on the Intel hardware much more efficient, especially on the CPU. Also, the resulting IR model is required to run PTQ optimization. IR model consists of 2 files: ``exported_model.xml`` for weights and ``exported_model.bin`` for architecture.

2. We can run the below command line to export the trained model
and save the exported model to the ``openvino`` folder:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx export --work_dir otx-workspace
            ...
            Elapsed time: 0:00:06.588245

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx export ... --checkpoint otx-workspace/20240313_042421/checkpoints/epoch_010.ckpt
            ...
            Elapsed time: 0:00:06.588245

    .. tab-item:: API

        .. code-block:: python

            engine.export()

Now that we have the exported model, let's check its performance using ``otx test``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \
                                --checkpoint otx-workspace/20240313_052847/exported_model.xml \
                                --engine.device cpu
            ...

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config src/otx/recipe/anomaly_detection/padim.yamll \
                                --data_root data/wgisd \
                                --checkpoint otx-workspace/20240312_052847/exported_model.xml \
                                --engine.device cpu
            ...

    .. tab-item:: API

        .. code-block:: python

            exported_model = engine.export()
            engine.test(checkpoint=exported_model)


************
Optimization
************

Anomaly tasks can be optimized either in PTQ or NNCF format. The model will be quantized to ``INT8`` format.
For more information refer to the :doc:`optimization explanation <../../../explanation/additional_features/models_optimization>` section.


1. Let's start with PTQ
optimization.

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

Please note, that PTQ will take some time without logging to optimize the model.

3. Finally, we can also evaluate the optimized model by passing
it to the ``otx test`` function.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace \ 
                                --checkpoint otx-workspace/20240313_055042/optimized_model.xml \
                                --engine.device cpu

            ...
            Elapsed time: 0:00:10.260521

    .. tab-item:: API

        .. code-block:: python

            ckpt_path = "otx-workspace/20240313_055042/optimized_model.xml"
            engine.test(checkpoint=ckpt_path)


*******************************
Segmentation and Classification
*******************************

While the above example shows Anomaly Detection, you can also train Anomaly Segmentation and Classification models.
To see what tasks are available, you can pass ``ANOMALY_SEGMENTATION`` and ``ANOMALY_CLASSIFICATION`` to ``otx find`` mentioned in the `Training`_ section. You can then use the same commands to train, evaluate, export and optimize the models.

.. note::

    The Segmentation and Detection tasks also require that the ``ground_truth`` masks be present to ensure that the localization metrics are computed correctly.
    The ``ground_truth`` masks are not required for the Classification task.

