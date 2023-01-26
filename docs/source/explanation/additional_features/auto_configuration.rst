Auto-configuration
==================

Auto-configuration for a deep learning framework means the automatic finding of the most appropriate settings for the training parameters, based on the dataset and the specific task at hand.
Auto-configuration can help to save time, ease the process of interaction with OTX CLI and give a better baseline for the given dataset.

At this end, we developed a simple auto-configuration functionality to ease the process of training and validation utilizing our framework.
Basically, to start the training and obtain a good baseline with the best trade-off between accuracy and speed we need to pass only a dataset in the right format without specifiing anything else:

.. code-block::

    $ otx train --train-data-root <path_to_data_root>

Moreover, our dataset can have no train/val splits at all. The Datumaro manager integrated into OTX will handle it on its own.
It will recognize the task by analizing the dataset and if there is no splits for the validation - Datumaro will do random auto-split, memorizing this split to further use it with ``otx eval`` and ``otx optimize``.
Currently, Datumaro auto-split feature supports 3 tasks: **multi-class classification**, **detection** and **semantic segmentation**.
After dataset preparation the training will be started with the middle-sized template to achieve a competitive accuracy preserving fast inference.

Supported dataset formats for each task:

- classificaiton: Imagenet, COCO (multi-label), :ref:`custom hierarchical <hierarchical_dataset>`
- object detection: COCO, Pascal-VOC, YOLO
- semantic segmentation: common semantic segmentation, Pascal-VOC, cityscapes, ADE20k
- action classification: CVAT
- action : CVAT
- anomaly classificaiton: MVTec
- anomaly detection: MVTec
- anomaly segmentation: MVTec
- instance segmentation: COCO, Pascal-VOC

.. note::

    If we have a dataset format that occluded with other tasks, for example ``COCO`` format, we should directly emphasize the task type with an additional CLI option:

    .. code-block::

        $ otx train --train-data-root <path_to_data_root> --task {CLASSIFICATION, DETECTION, SEGMENTATION, ACTION_CLASSIFICATION, ACTION_DETECTION, ANOMALY_CLASSIFICATION, ANOMALY_DETECTION, ANOMALY_SEGMENTATION, INSTANCE_SEGMENTATION}
