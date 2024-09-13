Configurable Input Size
=======================

The Configurable Input Size feature allows users to adjust the input resolution of their deep learning models
to balance between training and inference speed and model performance.
This flexibility enables users to tailor the input size to their specific needs without manually altering
the data pipeline configurations.

To utilize this feature, simply specify the desired input size as an argument during the train command.
Additionally, OTX ensures compatibility with model trained on non-default input sizes by automatically adjusting
the data pipeline to match the input size during other engine entry points.

Usage example:

.. code-block::

    $ otx train \
        --config ... \

.. tab-set::

    .. tab-item:: API 1

        .. code-block:: python

            from otx.algo.detection.yolox import YOLOXS
            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            input_size = (512, 512)
            model = YOLOXS(label_info=5, input_size=input_size)  # should be tuple[int, int]
            datamodule = OTXDataModule(..., input_size=input_size)
            engine = Engine(model=model, datamodule=datamodule)
            engine.train()

    .. tab-item:: API 2

        .. code-block:: python

            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            datamodule = OTXDataModule(..., input_size=(512, 512))
            engine = Engine(model="yolox_s", datamodule=datamodule)  # model input size will be aligned with the datamodule input size
            engine.train()

    .. tab-item:: CLI

        .. code-block:: bash

            (otx) ...$ otx train ... --data.input_size 512

.. _adaptive-input-size:

Adaptive Input Size
-------------------

The Adaptive Input Size feature intelligently determines an optimal input size for the model
by analyzing the dataset's statistics.
It operates in two distinct modes: "auto" and "downscale".
In "auto" mode, the input size may increase or decrease based on the dataset's characteristics.
In "downscale" mode, the input size will either decrease or remain unchanged, ensuring that the model training or inference speed deosn't drop.


To activate this feature, use the following command with the desired mode:

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.detection.yolox import YOLOXS
            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            datamodule = OTXDataModule(
                ...
                adaptive_input_size="auto",  # auto or downscale
                input_size_multiplier=YOLOXS.input_size_multiplier, # should set the input_size_multiplier of the model
            )
            model = YOLOXS(label_info=5, input_size=datamodule.input_size)
            engine = Engine(model=model, datamodule=datamodule)
            engine.train()

    .. tab-item:: CLI

        .. code-block:: bash

            (otx) ...$ otx train ... --data.adaptive_input_size "auto | downscale"

The adaptive process includes the following steps:

1. OTX computes robust statistics from the input dataset.

2. The initial input size is set based on the typical large image size within the dataset.

3. (Optional) The input size may be further refined to account for the sizes of objects present in the dataset.
   The model's minimum recognizable object size, typically ranging from 16x16 to 32x32 pixels, serves as a reference to
   proportionally adjust the input size relative to the average small object size observed in the dataset.
   For instance, if objects are generally 64x64 pixels in a 512x512 image, the input size would be adjusted
   to 256x256 to maintain detectability.

   Adjustments are subject to the following constraints:

   * If the recalculated input size exceeds the maximum image size determined in the previous step, it will be capped at that maximum size.
   * If the recalculated input size falls below the minimum threshold defined by MIN_DETECTION_INPUT_SIZE, the input size will be scaled up. This is done by increasing the smaller dimension (width or height) to MIN_DETECTION_INPUT_SIZE while maintaining the aspect ratio, ensuring that the model's minimum criteria for object detection are met.

4. (downscale only) Any scale-up beyond the default model input size is restricted.


.. Note::
    Opting for a smaller input size can be advantageous for datasets with lower-resolution images or larger objects,
    as it may improve speed with minimal impact on model accuracy. However, it is important to consider that selecting
    a smaller input size could affect model performance depending on the task, model architecture, and dataset
    properties.
