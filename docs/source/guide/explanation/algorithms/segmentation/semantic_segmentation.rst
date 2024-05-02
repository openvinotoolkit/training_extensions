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

We solve this task by utilizing segmentation decoder heads on the multi-level image features obtained by the feature extractor backbone.
For the supervised training we use the following algorithms components:

.. _semantic_segmentation_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip, random rotate and random crop, we use mixing images technique with different `photometric distortions <https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.PhotoMetricDistortion>`_.

- ``Optimizer``: We use `Adam <https://arxiv.org/abs/1412.6980>`_ and `AdamW <https://arxiv.org/abs/1711.05101>` optimizers.

- ``Learning rate schedule``: For scheduling training process we use **ReduceLROnPlateau** with linear learning rate warmup for 100 iterations for `Lite-HRNet <https://arxiv.org/abs/2104.06403>`_ family. This method monitors a target metric (in our case we use metric on the validation set) and if no improvement is seen for a ``patience`` number of epochs, the learning rate is reduced.
    For `SegNext <https://arxiv.org/abs/2209.08575>`_ and `DinoV2 <https://arxiv.org/abs/2304.07193>`_ models we use `PolynomialLR <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.PolynomialLR.html>`_ scheduler.

- ``Loss function``: We use standard `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting.

**************
Dataset Format
**************

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

At this end we support `Common Semantic Segmentation <https://github.com/openvinotoolkit/datumaro/blob/develop/docs/source/docs/data-formats/formats/common_semantic_segmentation.md>`_ data format.
If you organized supported dataset format, starting training will be very simple. We just need to pass a path to the root folder and desired model recipe to start training:


******
Models
******
.. _semantic_segmentation_models:

We support the following ready-to-use model recipes:

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| Recipe Path                                                                                                                                                                          | Complexity (GFLOPs) | Model size (M)  | FPS (GPU)       | iter time (sec) |
+======================================================================================================================================================================================+=====================+=================+=================+=================+
| `Lite-HRNet-s-mod2 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_s.yaml>`_                                     | 1.44                | 0.82            |  37.68          |     0.151       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `Lite-HRNet-18-mod2 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_18.yaml>`_                                   | 2.63                | 1.10            |  31.17          |     0.176       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `Lite-HRNet-x-mod3 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/litehrnet_x.yaml>`_                                     | 9.20                | 1.50            |  15.07          |     0.347       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `SegNext_T <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_t.yaml>`_                                               | 12.44               | 4.23            |  104.90         |     0.126       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `SegNext_S <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_s.yaml>`_                                               | 30.93               | 13.90           |  85.67          |     0.134       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `SegNext_B <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/segnext_b.yaml>`_                                               | 64.65               | 27.56           |  61.91          |     0.215       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+
| `DinoV2 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/semantic_segmentation/dino_v2.yaml>`_                                                    | 124.01              | 24.40           |  3.52           |     0.116       |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+-----------------+-----------------+

All of these models differ in the trade-off between accuracy and inference/training speed. For example, ``SegNext_B`` is the recipe with heavy-size architecture for more accurate predictions, but it requires longer training.
Whereas the ``Lite-HRNet-s-mod2`` is the lightweight architecture for fast inference and training. It is the best choice for the scenario of a limited amount of data. The ``Lite-HRNet-18-mod2`` and ``SegNext_S``  models are the middle-sized architectures for the balance between fast inference and training time.
``DinoV2`` is the state-of-the-art model producing universal features suitable for all image-level and pixel-level visual tasks. This model doesn't require fine-tuning of the whole backbone, but only segmentation decode head. Because of that, it provides faster training preserving high accuracy.

In the table below the `Dice score <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ on some academic datasets using our :ref:`supervised pipeline <semantic_segmentation_supervised_pipeline>` is presented. We use 512x512 (560x560 fot DinoV2) image crop resolution, for other hyperparameters, please, refer to the related recipe. We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Model name            | `DIS5K <https://xuebinqin.github.io/dis/index.html>`_        | `Cityscapes <https://www.cityscapes-dataset.com/>`_ | `Pascal-VOC 2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ | `KITTI <https://www.cvlibs.net/datasets/kitti/index.php>`_      | Mean   |
+=======================+==============================================================+=====================================================+======================================================================+=================================================================+========+
| Lite-HRNet-s-mod2     | 78.73                                                        | 69.25                                               | 63.26                                                                | 41.73                                                           | 63.24  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Lite-HRNet-18-mod2    | 81.43                                                        | 72.66                                               | 62.10                                                                | 46.73                                                           | 65.73  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Lite-HRNet-x-mod3     | 82.36                                                        | 74.57                                               | 59.55                                                                | 49.97                                                           | 66.61  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-t             | 83.99                                                        | 77.09                                               | 84.05                                                                | 48.99                                                           | 73.53  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-s             | 85.54                                                        | 79.45                                               | 86.00                                                                | 52.19                                                           | 75.80  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| SegNext-b             | 86.76                                                        | 76.14                                               | 87.92                                                                | 57.73                                                           | 77.14  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| DinoV2                | 84.87                                                        | 73.58                                               | 88.15                                                                | 65.91                                                           | 78.13  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/semantic_segmentation>` for more information on how to train, validate and optimize the semantic segmentation model.
