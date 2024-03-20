OpenVINO™ Training Extensions API Quick-Start
==============================================

Besides CLI functionality, The OpenVINO™ Training Extension provides APIs that help developers to integrate OpenVINO™ Training Extensions models into their projects.
This tutorial intends to show how to create a dataset, model and use all of the CLI functionality through APIs.

For demonstration purposes we will use the Object Detection SSD model with `WGISD <https://github.com/thsant/wgisd>`_ public dataset as we did for the :doc:`CLI tutorial <../tutorials/base/how_to_train/detection>`.

.. note::

    To start with we need to `install OpenVINO™ Training Extensions <https://github.com/openvinotoolkit/training_extensions/blob/develop/QUICK_START_GUIDE.md#setup-openvino-training-extensions>`_.

*******************
Dataset preparation
*******************

1. Clone a repository
with `WGISD dataset <https://github.com/thsant/wgisd>`_.

.. code-block:: shell

  cd data
  git clone https://github.com/thsant/wgisd.git
  cd wgisd
  git checkout 6910edc5ae3aae8c20062941b1641821f0c30127

2. We need to rename annotations to
be distinguished by OpenVINO™ Training Extensions Datumaro manager:

.. code-block:: shell

    mv data images && mv coco_annotations annotations && mv annotations/train_bbox_instances.json instances_train.json  && mv annotations/test_bbox_instances.json instances_val.json

Now it is all set to use this dataset inside OpenVINO™ Training Extensions

************************************
Quick Start with auto-configuration
************************************

Once the dataset is ready, we can immediately start training with the model and data pipeline recommended by OTX through auto-configuration.
The following code snippet demonstrates how to use the auto-configuration feature:

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(data_root="data/wgisd")
    engine.train()


.. note::

    If dataset supports multiple Task types, this will default to the Task type detected by OTX.
    If you want to specify a specific Task type, you need to specify it like below:

    .. code-block:: python

        from otx.engine import Engine

        engine = Engine(data_root="data/wgisd", task="INSTANCE_SEGMENTATION")
        engine.train()


**********************************
Check Available Model Recipes
**********************************

If you want to use other models offered by OTX besides the ones provided by Auto-Configuration, you can get a list of available models in OTX as shown below.

.. code-block:: python

    from otx.engine.utils.api import list_models

    model_lists = list_models(task="DETECTION")
    print(model_lists)

    '''
    [
        'yolox_tiny_tile',
        'yolox_x',
        'yolox_l_tile',
        'yolox_x_tile', 'yolox_l',
        'atss_r50_fpn',
        'ssd_mobilenetv2',
        'yolox_s',
        'yolox_tiny',
        'openvino_model',
        'atss_mobilenetv2',
        'yolox_s_tile',
        'rtmdet_tiny',
        'atss_mobilenetv2_tile',
        'atss_resnext101',
        'ssd_mobilenetv2_tile',
    ]
    '''


.. note::

    If you're looking for a specific name, use the pattern argument.

    .. code-block:: python

        from otx.engine.utils.api import list_models

        model_lists = list_models(task="DETECTION", pattern="tile")
        print(model_lists)
        '''
        [
            'yolox_tiny_tile',
            'ssd_mobilenetv2_tile',
            'yolox_l_tile',
            'yolox_s_tile',
            'yolox_x_tile',
            'atss_mobilenetv2_tile',
        ]
        '''


You can also find the available model recipes in YAML form in the folder ``otx/recipe``.

*********
Engine
*********

The ``otx.engine.Engine`` class is the main entry point for using OpenVINO™ Training Extensions APIs.

1. Setting ``task``

Specify ``task``. This is the task type for that ``Engine`` usage.
You can set the task by referencing the ``OTXTaskType`` in ``otx.core.types.task``.
If no task is specified, the task is detected and used via ``datamodule`` or ``data_root``.

.. code-block:: python

    from otx.core.types.task import OTXTaskType
    from otx.engine import Engine

    engine = Engine(task=OTXTaskType.DETECTION)
    # or
    engine = Engine(task="DETECTION")

2. Setting ``work_dir``

Specify ``work_dir``. This is the workspace for that ``Engine``, and where output is stored.
The default value is currently ``./otx-workspace``.

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(work_dir="work_dir")


3. Setting device

You can set the device by referencing the ``DeviceType`` in ``otx.core.types.device``.
The current default setting is ``auto``.

.. code-block:: python

    from otx.core.types.device import DeviceType
    from otx.engine import Engine

    engine = Engine(device=DeviceType.gpu)
    # or
    engine = Engine(device="gpu")


In addition, the ``Engine`` constructor can be associated with the Trainer's constructor arguments to control the Trainer's functionality.
Refer `lightning.Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

4. Using the OTX configuration we can configure the Engine.

.. code-block:: python

    from otx.engine import Engine

    recipe = "src/otx/recipe/detection/atss_mobilenetv2.yaml"
    engine = Engine.from_config(
        config_path=recipe,
        data_root="data/wgisd",
        work_dir="./otx-workspace",
    )


*********
Training
*********

Create an output model and start actual training:

1. Below is an example using the ``atss_mobilenetv2`` model provided by OTX.

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(data_root="data/wgisd", model="atss_mobilenetv2")
    engine.train()

2. Alternatively, we can use the configuration file.

.. code-block:: python

    from otx.engine import Engine

    config = "src/otx/recipe/detection/atss_mobilenetv2.yaml"

    engine = Engine.from_config(config_path=config, data_root="data/wgisd")
    engine.train()

.. note::

    This can use callbacks provided by OTX and several training techniques.
    However, in this case, no arguments are specified for train.

3. If you want to specify the model, you can do so as shown below:

The model used by the Engine is of type ``otx.core.model.entity.base.OTXModel``.

.. tab-set::

    .. tab-item:: Custom Model

        .. code-block:: python

            from otx.algo.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(num_classes=5, variant="mobilenetv2")

            engine = Engine(data_root="data/wgisd", model=model)
            engine.train()

    .. tab-item:: Custom Model with checkpoint

        .. code-block:: python

            from otx.algo.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(num_classes=5, variant="mobilenetv2")

            engine = Engine(data_root="data/wgisd", model=model, checkpoint="<path/to/checkpoint>")
            engine.train()

    .. tab-item:: Custom Optimizer & Scheduler

        .. code-block:: python

            from torch.optim import SGD
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from otx.algo.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(num_classes=5, variant="mobilenetv2")
            optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)

            engine = Engine(
                ...,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            engine.train()

4. If you want to specify the datamodule, you can do so as shown below:

The datamodule used by the Engine is of type ``otx.core.data.module.OTXDataModule``.

.. code-block:: python

    from otx.core.data.module import OTXDataModule
    from otx.engine import Engine

    datamodule = OTXDataModule(data_root="data/wgisd")

    engine = Engine(datamodule=datamodule)
    engine.train()

.. note::

    If both ``data_root`` and ``datamodule`` enter ``Engine`` as input, ``Engine`` uses datamodule as the base.


5. You can use train-specific arguments with ``train()`` function.

.. tab-set::

    .. tab-item:: Change Max Epochs

        .. code-block:: python

            engine.train(max_epochs=10)

    .. tab-item:: Fix Training Seed & Set Deterministic

        .. code-block:: python

            engine.train(seed=1234, deterministic=True)

    .. tab-item:: Use Mixed Precision

        .. code-block:: python

            engine.train(precision="16")

        .. note::
            
            This uses lightning's precision value. You can use the values below:
            - "64", "32", "16", "bf16",
            - 64, 32, 16

    .. tab-item:: Change Validation Metric

        .. code-block:: python

            from otx.core.metrics.fmeasure import FMeasure

            metric = FMeasue(num_classes=5)
            engine.train(metric=metric)

    .. tab-item:: Set Callbacks & Logger

        .. code-block:: python

            from lightning.pytorch.callbacks import EarlyStopping
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

            engine.train(callbacks=[EarlyStopping()], loggers=[TensorBoardLogger()])

In addition, the ``train()`` function can be associated with the Trainer's constructor arguments to control the Trainer's functionality.
Refer `lightning.Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

For example, if you want to use the ``limit_val_batches`` feature provided by Trainer, you can use it like this:

.. code-block:: python

    # disable validation
    engine.train(limit_val_batches=0)

6. It's also easy to use HPOs.

.. code-block:: python

    engine.train(run_hpo=True)


***********
Evaluation
***********

If the training is already in place, we just need to use the code below:

.. tab-set::

    .. tab-item:: Evaluate Model

        .. code-block:: python

            engine.test()

    .. tab-item:: Evaluate Model with different checkpoint

        .. code-block:: python

            engine.test(checkpoint="<path/to/checkpoint>")

        .. note::

            The format that can enter the checkpoint is of type torch (.ckpt) or exported model (.onnx, .xml).

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.core.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.test(datamodule=datamodule)

    .. tab-item:: Evaluate Model with different metrics

        .. code-block:: python

            from otx.core.metrics.fmeasure import FMeasure

            metric = FMeasue(num_classes=5)
            engine.test(metric=metric)


***********
Exporting
***********

To export our model to OpenVINO™ IR format we need to create output model and run exporting task.
If the engine is trained, you can proceed with the export using the current engine's checkpoint:

The default value for ``export_format`` is ``OPENVINO``.
The default value for ``export_precision`` is ``FP32``.

.. tab-set::

    .. tab-item:: Export OpenVINO™ IR

        .. code-block:: python

            engine.export()

    .. tab-item:: Export ONNX

        .. code-block:: python

            engine.export(export_format="ONNX")

    .. tab-item:: Export with explain features

        .. code-block:: python

            engine.export(explain=True)

        .. note::

            This ensures that it is exported with a ``saliency_map`` and ``feature_vector`` that will be used in the XAI.

    .. tab-item:: Export with different checkpoint

        .. code-block:: python

            engine.export(checkpoint="<path/to/checkpoint>")

    .. tab-item:: Export with FP16

        .. code-block:: python

            engine.export(precision="FP16")


****
XAI
****

To run the XAI with the OpenVINO™ IR model, we need to create an output model and run the XAI procedure:

.. tab-set::

    .. tab-item:: Run XAI

        .. code-block:: python

            engine.explain(checkpoint="<path/to/ir/xml>")

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.core.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.explain(..., datamodule=datamodule)

    .. tab-item:: Dump saliency_map

        .. code-block:: python

            engine.explain(..., dump=True)


************
Optimization
************

To run the optimization with PTQ on the OpenVINO™ IR model, we need to create an output model and run the optimization procedure:

.. tab-set::

    .. tab-item:: Run PTQ Optimization

        .. code-block:: python

            engine.optimize(checkpoint="<path/to/ir/xml>")

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.core.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.optimize(..., datamodule=datamodule)


You can validate the optimized model as the usual model. For example for the NNCF model it will look like this:

.. code-block:: python

    engine.test(checkpoint="<path/to/optimized/ir/xml>")

That's it. Now, we can use OpenVINO™ Training Extensions APIs to create, train, and deploy deep learning models using the OpenVINO™ Training Extension.
