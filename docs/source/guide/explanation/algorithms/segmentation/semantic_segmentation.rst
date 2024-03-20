Semantic Segmentation
=====================

Semantic segmentation is a computer vision task in which an algorithm assigns a label or class to each pixel in an image.
For example, semantic segmentation can be used to identify the boundaries of different objects in an image, such as cars, buildings, and trees.
The output of semantic segmentation is typically an image where each pixel is colored with a different color or label depending on its class.

.. _semantic_segmentation_image_example:


.. image:: ../../../../../utils/images/semantic_seg_example.png
  :width: 600
  :alt: image uploaded from this `source <https://arxiv.org/abs/1912.03183>`_

|

We solve this task by utilizing `FCN Head <https://arxiv.org/pdf/1411.4038.pdf>`_ with implementation from `MMSegmentation <https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/models/decode_heads/fcn_head.html>`_ on the multi-level image features obtained by the feature extractor backbone (`Lite-HRNet <https://arxiv.org/abs/2104.06403>`_).
For the supervised training we use the following algorithms components:

.. _semantic_segmentation_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip, random rotate and random crop, we use mixing images technique with different `photometric distortions <https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.PhotoMetricDistortion>`_.

- ``Optimizer``: We use `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer with weight decay set to zero and gradient clipping with maximum quadratic norm equals to 40.

- ``Learning rate schedule``: For scheduling training process we use **ReduceLROnPlateau** with linear learning rate warmup for 100 iterations. This method monitors a target metric (in our case we use metric on the validation set) and if no improvement is seen for a ``patience`` number of epochs, the learning rate is reduced.

- ``Loss function``: We use standard `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting.

**************
Dataset Format
**************

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

At this end we support `Common Semantic Segmentation <https://github.com/openvinotoolkit/datumaro/blob/develop/docs/source/docs/data-formats/formats/common_semantic_segmentation.md>`_ data format.
If you organized supported dataset format, starting training will be very simple. We just need to pass a path to the root folder and desired model template to start training:

.. note::

    Due to some internal limitations, the dataset should always consist of a "background" label. If your dataset doesn't have a background label, rename the first label to "background" in the ``meta.json`` file.


.. note::

    Currently, metrics with models trained with our OTX dataset adapter can differ from popular benchmarks. To avoid this and train the model on exactly the same segmentation masks as intended by the authors, please, set the parameter ``use_otx_adapter`` to ``False``.

******
Models
******
.. _semantic_segmentation_models:

We support the following ready-to-use model templates:

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                          | Name                   | Complexity (GFLOPs) | Model size (MB) |
+======================================================================================================================================================================================+========================+=====================+=================+
| `Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_s.yaml>`_    | Lite-HRNet-s-mod2      | 1.44                | 3.2             |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_18.yaml>`_  | Lite-HRNet-18-mod2     | 2.82                | 4.3             |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_x.yaml>`_    | Lite-HRNet-x-mod3      | 9.20                | 5.7             |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_SegNext_T <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_t.yaml>`_                  | SegNext-t              | 6.07                | 4.23            |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_SegNext_S <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_s.yaml>`_                  | SegNext-s              | 15.35               | 13.9            |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_SegNext_B <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_b.yaml>`_                  | SegNext-b              |   32.08             | 27.56           |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

All of these models are members of the same `Lite-HRNet <https://arxiv.org/abs/2104.06403>`_ backbones family. They differ in the trade-off between accuracy and inference/training speed. ``Lite-HRNet-x-mod3`` is the template with heavy-size architecture for accurate predictions but it requires long training.
Whereas the ``Lite-HRNet-s-mod2`` is the lightweight architecture for fast inference and training. It is the best choice for the scenario of a limited amount of data. The ``Lite-HRNet-18-mod2`` model is the middle-sized architecture for the balance between fast inference and training time.

Use `SegNext <https://arxiv.org/abs/2209.08575>`_ model which can achieve superior perfomance while preserving fast inference and fast training.

In the table below the `Dice score <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ on some academic datasets using our :ref:`supervised pipeline <semantic_segmentation_supervised_pipeline>` is presented. We use 512x512 image crop resolution, for other hyperparameters, please, refer to the related template. We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Model name            | `DIS5K <https://xuebinqin.github.io/dis/index.html>`_        | `Cityscapes <https://www.cityscapes-dataset.com/>`_ | `Pascal-VOC 2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ | `KITTI full <https://www.cvlibs.net/datasets/kitti/index.php>`_ | Mean   |
+=======================+==============================================================+=====================================================+======================================================================+=================================================================+========+
| Lite-HRNet-s-mod2     | 79.95                                                        | 62.38                                               | 58.26                                                                | 36.06                                                           | 59.16  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Lite-HRNet-18-mod2    | 81.12                                                        | 65.04                                               | 63.48                                                                | 39.14                                                           | 62.20  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Lite-HRNet-x-mod3     | 79.98                                                        | 59.97                                               | 61.9                                                                 | 41.55                                                           | 60.85  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-t             | 85.05                                                        | 70.67                                               | 80.73                                                                | 51.25                                                           | 68.99  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-s             | 85.62                                                        | 70.91                                               | 82.31                                                                | 52.94                                                           | 69.82  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-b             | 87.92                                                        | 76.94                                               | 85.01                                                                | 55.49                                                           | 73.45  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/semantic_segmentation>` for more information on how to train, validate and optimize the semantic segmentation model.
