Auto-configuration
==================

Auto-configuration for a deep learning framework means the automatic finding of the most appropriate settings for the training parameters, based on the dataset and the specific task at hand.
Auto-configuration can help to save time, it eases the process of interaction with OpenVINO™ Training Extensions CLI and gives a better baseline for the given dataset.

At this end, we developed a simple auto-configuration functionality to ease the process of training and validation utilizing our framework.
Basically, to start the training and obtain a good baseline with the best trade-off between accuracy and speed we need to pass only a dataset in the right format without specifying anything else:

.. code-block::

    $ otx train --train-data-roots <path_to_data_root>

.. note::

    OpenVINO™ Training Extensions supports also ``otx build`` mode with the auto-configuration feature. We can build OpenVINO™ Training Extensions workspace with the following CLI command:

    .. code-block::

        $ otx build --train-data-roots <path_to_data_root>

Moreover, our dataset can have no train/val splits at all. The Datumaro manager integrated into OpenVINO™ Training Extensions will handle it on its own.
It will recognize the task by analyzing the dataset and if there is no splits for the validation - Datumaro will do a random auto-split, saving this split to the workspace. It could be used with ``otx optimize`` or ``otx train``.

.. note::

    Currently, Datumaro auto-split feature supports 3 formats: `Imagenet <https://www.image-net.org/>`_  (multi-class classification), `COCO <https://cocodataset.org/#format-data>`_ (detection) and `Cityscapes <https://openvinotoolkit.github.io/datumaro/docs/formats/cityscapes/>`_ (semantic segmentation).

After dataset preparation, the training will be started with the middle-sized template to achieve competitive accuracy preserving fast inference.


Supported dataset formats for each task:

- classification: `Imagenet <https://www.image-net.org/>`_, `COCO <https://cocodataset.org/#format-data>`_ (multi-label), :ref:`custom hierarchical <hierarchical_dataset>`
- object detection: `COCO <https://cocodataset.org/#format-data>`_, `Pascal-VOC <https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/>`_, `YOLO <https://openvinotoolkit.github.io/datumaro/docs/formats/yolo/>`_
- semantic segmentation: `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/docs/formats/common_semantic_segmentation/>`_, `Pascal-VOC <https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/>`_, `Cityscapes <https://openvinotoolkit.github.io/datumaro/docs/formats/cityscapes/>`_, `ADE20k <https://openvinotoolkit.github.io/datumaro/docs/formats/ade20k2020/>`_
- action classification: `CVAT <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_
- action detection: `CVAT <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_
- anomaly classification: `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_
- anomaly detection: `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_
- anomaly segmentation: `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_
- instance segmentation: `COCO <https://cocodataset.org/#format-data>`_, `Pascal-VOC <https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/>`_

If we have a dataset format occluded with other tasks, for example ``COCO`` format, we should directly emphasize the task type and use ``otx build`` first with an additional CLI option. If not, OpenVINO™ Training Extensions automatically chooses the task type that you might not intend:

.. code-block::

    $ otx build --train-data-roots <path_to_data_root>
                --task {CLASSIFICATION, DETECTION, SEGMENTATION, ACTION_CLASSIFICATION, ACTION_DETECTION, ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION, INSTANCE_SEGMENTATION}

It will create a task-specific workspace folder with configured template and auto dataset split if supported.

Move to this folder and simply run without any options to start training:

.. code-block::

    $ otx train


Auto-adapt batch size
---------------------

This feature adapts a batch size based on the current hardware environment.
There are two methods available for adapting the batch size.

1. Prevent GPU Out of Memory (`Safe` mode)

The first method checks if the current batch size is compatible with the available GPU devices.
Larger batch sizes consume more GPU memory for training. Therefore, the system verifies if training is possible with the current batch size.
If it's not feasible, the batch size is decreased to reduce GPU memory usage.
However, setting the batch size too low can slow down training.
To address this, the batch size is reduced to the maximum amount that could be run safely on the current GPU resource.
The learning rate is also adjusted based on the updated batch size accordingly.

To use this feature, add the following parameter:

.. code-block::

    $ otx train params --learning_parameters.auto_adapt_batch_size Safe

2. Find the maximum executable batch size (`Full` mode)

The second method aims to find a possible large batch size that reduces the overall training time.
Increasing the batch size reduces the effective number of iterations required to sweep the whole dataset, thus speeds up the end-to-end training.
However, it does not search for the maximum batch size as it is not efficient and may require significantly more time without providing substantial acceleration compared to a large batch size.
Similar to the previous method, the learning rate is adjusted according to the updated batch size accordingly.

To use this feature, add the following parameter:

.. code-block::

    $ otx train params --learning_parameters.auto_adapt_batch_size Full


.. Warning::
    When using a fixed epoch, training with larger batch sizes is generally faster than with smaller batch sizes.
    However, if early stop is enabled, training with a lower batch size can finish early.


Auto-adapt num_workers
----------------------

This feature adapts the ``num_workers`` parameter based on the current hardware environment.
The ``num_workers`` parameter controls the number of subprocesses used for data loading during training.
While increasing ``num_workers`` can reduce data loading time, setting it too high can consume a significant amount of CPU memory.

To simplify the process of setting ``num_workers`` manually, this feature automatically determines the optimal value based on the current hardware status.

To use this feature, add the following parameter:

.. code-block::

    $ otx train params --learning_parameters.auto_num_workers True

Auto-detect training type
-------------------------

OpenVINO™ Training Extensions also support automatic detection of training types such as Semi-SL, Self-SL and Incremental. For Semi-SL usage only is a path to unlabeled data via `--unlabeled-data-roots` option needed for the command line. To use Self-SL learning just a folder with images in the `--train-data-roots` option without validation data is required to automatically start Self-SL pretraining.
OpenVINO™ Training Extensions will automatically recognize these types of tasks and if the task supports this training type the training will be started.

.. note::
    To use auto template configuration with Self-SL training type `--task` option is required since it is impossible to recognize task type by folder with only images.
