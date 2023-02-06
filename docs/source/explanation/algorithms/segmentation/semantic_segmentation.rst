Semantic Segmentation
=====================

Semantic segmentation is a computer vision task in which an algorithm assigns a label or class to each pixel in an image.
For example, semantic segmentation can be used to identify the boundaries of different objects in an image, such as cars, buildings, and trees.
The output of semantic segmentation is typically an image where each pixel is colored with a different color or label depending on its class.

.. _semantic_segmentation_image_example:

.. image:: ../../../../utils/images/semantic_seg_example.png
  :width: 600
  :alt: image uploaded from this `source <https://arxiv.org/abs/1912.03183>`_


We solve this task by utilizing `FCN Head <https://arxiv.org/pdf/1411.4038.pdf>`_ with implementation from `MMSegmentation <https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/models/decode_heads/fcn_head.html>`_ on the multi-level image features obtained by the feature extractor backbone (`Lite-HRNet <https://arxiv.org/abs/2104.06403>`_).
For the supervised training we use the following algorithms components:

.. _semantic_segmentation_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip, random rotate and random crop, we use mixing images technique with different `photometric distortions <https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.datasets.pipelines.PhotoMetricDistortion>`_.

- ``Optimizer``: We use `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer with weight decay set to zero and gradient clipping with maximum quadratic norm equals to 40.

- ``Learning rate schedule``: For scheduling training process we use **ReduceLROnPlataeu** with linear learning rate warmup for 100 iterations. This method monitors a target metric (in our case we use metric on the validation set) and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

- ``Loss function``: We use standart `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model.

- Additionally, we use the **early stopping** to add adaptability to the training pipeline and prevent overfitting.

**************
Dataset Format
**************

For the dataset handling inside OTX we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

At this end we support `ADE20K <https://openvinotoolkit.github.io/datumaro/docs/formats/ade20k2020/>`_, `Cityscapes <https://openvinotoolkit.github.io/datumaro/docs/formats/cityscapes/>`__, `Pascal VOC <https://openvinotoolkit.github.io/datumaro/docs/formats/pascal_voc/>`_ and `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/docs/formats/common_semantic_segmentation/>`_ data formats.
If you organized supported dataset format, starting training will be very simple. We just need to pass a path to the root folder and desired model template to start training:

.. code-block::

    $ otx train --template <model_template> --train-data-root <path_to_data_root> --val-data-root <path_to_data_root>

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/semantic_segmentation>` for more information on how to train, validate and optimize semantic segmentation model for more details.

******
Models
******

We support the following ready-to-use model templates:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                                                                  | Name                   | Complexity (GFLOPs) | Model size (MB) |
+==============================================================================================================================================================================================================================+========================+=====================+=================+
| `Custom_Semantic_Segmentation_Lite-HRNet-s-mod2_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/segmentation/configs/ocr_lite_hrnet_s_mod2/template.yaml>`_                      | Lite-HRNet-s-mod2      | 1.44                | 3.2             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/template.yaml>`_                    | Lite-HRNet-18-mod2     | 2.82                | 4.3             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Semantic_Segmentation_Lite-HRNet-x-mod3_OCR <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/segmentation/configs/ocr_lite_hrnet_x_mod3/template.yaml>`_                      | Lite-HRNet-x-mod3      | 9.20                | 5.7             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

All of these models are members of the same `Lite-HRNet <https://arxiv.org/abs/2104.06403>`_ backbones family. They differ in the trade-off between accuracy and inference/training speed. ``Lite-HRNet-x-mod3`` is the template with heavy-size architecture for accurate predictions but it requires long training.
Whereas the ``Lite-HRNet-s-mod2`` is the lightweight architecture for fast inference and training. It is the best choice for the scenario of a limited amount of data. The ``Lite-HRNet-18-mod2`` model is the middle-sized architecture for the balance between fast inference and training time.

In the table below the `Dice score <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ on some academic datasets using our :ref:`supervised pipeline <semantic_segmentation_supervised_pipeline>` is presented. The results were obtained on our templates without any changes. We use 512x512 image crop resolution, for other hyperparameters, please, refer to the related template. We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+--------------+------------+-----------------+
| Model name            | ADE20k       | Cityscapes | Pascal-VOC 2012 |
+=======================+==============+============+=================+
| Lite-HRNet-s-mod2     | N/A          | N/A        | N/A             |
+-----------------------+--------------+------------+-----------------+
| Lite-HRNet-18-mod2    | N/A          | N/A        | N/A             |
+-----------------------+--------------+------------+-----------------+
| Lite-HRNet-x-mod3     | N/A          | N/A        | N/A             |
+-----------------------+--------------+------------+-----------------+

************************
Semi-supervised Learning
************************

To solve :ref:`Semi-supervised learning <semi_sl_explanation>` problem for the semantic segmentation we use the `Mean Teacher algorithm <https://arxiv.org/abs/1703.01780>`_. The basic idea of this approach is to use two models during training: a "student" model, which is the main model being trained, and a "teacher" model, which acts as a guide for the student model.
The student model is updated based on the ground truth annotations (for the labeled data) and pseudo-labels (for the unlabeled data) which are the predictions of the teacher model.
The teacher model is updated based on the moving average of the student model's parameters. So, we don't use backward loss propagation for the teacher model's parameters.
After training, only the student model is used for prediction.

We utilize the same core algorithm's parameters as for the :ref:`supervised pipeline <semantic_segmentation_supervised_pipeline>`. The main difference is to use of different augmentation pipelines for the labeled and unlabeled data.
We use only basic augmentations (random flip, random rotate, random crop) for the labeled data and stronger for the unlabeled (color distortion).
It helps with a better generalization and prevents unnecessary overfitting on the pseudo-labels generated by the teacher model.

In the table below the `Dice score <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ with our middle template on some datasets is presented. For comparison, we present the supervised baseline trained on the labeled data only.
The results were obtained on our templates without any changes. We use 512x512 image resolution, for other hyperparameters, please, refer to the `related template <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/template.yaml>`_. We trained each model with single Nvidia GeForce RTX3090.
For `Cityscapes <https://www.cityscapes-dataset.com/>`__ and `Pascal-VOC <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html>`_ we use splits with different ratios of labeled to unlabeled data like `here <https://github.com/charlesCXK/TorchSemiSeg>`_.
For the `DIS5K <https://xuebinqin.github.io/dis/index.html>`_ we prepared random splits for the train data with different ratios of labeled to unlabeled images. We use the validation set for testing purposes.

+-------------------------------------+--------------+------------+-----------------+
| Model name                          |  DIS5K       | Cityscapes | Pascal-VOC      |
+=====================================+==============+============+=================+
| Supervised Lite-HRNet-18-mod2 (1/8) | N/A          | N/A        | N/A             |
+-------------------------------------+--------------+------------+-----------------+
| Semi-SL Lite-HRNet-18-mod2 (1/8)    | N/A          | N/A        | N/A             |
+-------------------------------------+--------------+------------+-----------------+
| Supervised Lite-HRNet-18-mod2 (1/16)| N/A          | N/A        | N/A             |
+-------------------------------------+--------------+------------+-----------------+
| Semi-SL Lite-HRNet-18-mod2 (1/16)   | N/A          | N/A        | N/A             |
+-------------------------------------+--------------+------------+-----------------+

************************
Self-supervised Learning
************************

To be added soon

********************
Incremental Learning
********************

To be added soon