Instance Segmentation
=====================

Instance segmentation is a computer vision task that involves identifying and segmenting individual objects within an image.

It is a more advanced version of object detection, as it doesn't only detect the presence of an object in an image but also segments the object by creating a mask that separates it from the background. This allows getting more detailed information about the object, such as its shape and location, to be extracted.

Instance segmentation is commonly used in applications such as self-driving cars, robotics, and image-editing software.

.. _instance_segmentation_image_example:


.. image:: ../../../../../utils/images/instance_seg_example.png
  :width: 600

|

We integrate two types of instance segmentation architecture within OpenVINO™ Training Extensions:: `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_ and `RTMDet <https://arxiv.org/abs/2212.07784>`_.

Mask R-CNN, a widely adopted method, builds upon the Faster R-CNN architecture, known for its two-stage object detection mechanism. In the initial stage, it proposes regions of interest, while in the subsequent stage, it predicts the class and bounding box offsets for each proposal. Distinguishing itself, Mask R-CNN incorporates an additional branch dedicated to predicting object masks concurrently with the existing branches for bounding box regression and object classification.

On the other hand, RTMDet leverages the architecture of `RTMNet <https://arxiv.org/abs/2212.07784>`_, a lightweight, one-stage model designed for both object detection and instance segmentation tasks. RTMNet prioritizes efficiency, making it particularly suitable for **real-time applications**. RTMDet-Inst extends the capabilities of RTMNet to encompass instance segmentation by integrating a mask prediction branch.


For the supervised training we use the following algorithms components:

.. _instance_segmentation_supervised_pipeline:

- ``Augmentations``: We use only a random flip for both augmentations pipelines, train and validation.

- ``Optimizer``: We use `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer with the weight decay set to **1e-4** and momentum set to **0.9**.

- ``Learning rate schedule``: For scheduling training process we use **ReduceLrOnPlateau** with linear learning rate warmup for **200** iterations. This method monitors a target metric (in our case we use metric on the validation set) and if no improvement is seen for a ``patience`` number of epochs, the learning rate is reduced.

- ``Loss functions``: For the bounding box regression we use **L1 Loss** (the sum of the absolute differences between the ground truth value and the predicted value), `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_ for the categories classification and segmentation masks prediction.

- Additionally, we use the **Exponential Moving Average (EMA)** for the model's weights and the **early stopping** to add adaptability to the training pipeline and prevent overfitting.

**************
Dataset Format
**************

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_. For instance segmentation we support `COCO <https://cocodataset.org/#format-data>`_ dataset format.

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/instance_segmentation>` how to train, validate and optimize instance segmentation model for more details.

******
Models
******

We support the following ready-to-use model recipes:

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------+---------------------+-----------------+
| Model Recipe                                                                                                                                                                                                  | Name                       | Complexity (GFLOPs) | Model size (MB) |
+===============================================================================================================================================================================================================+============================+=====================+=================+
| `Instance Segmentation MaskRCNN EfficientNetB2B <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/instance_segmentation/maskrcnn_efficientnetb2b.yaml>`_                    | MaskRCNN-EfficientNetB2B   | 68.48               | 13.27           |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------+---------------------+-----------------+
| `Instance Segmentation MaskRCNN ResNet50 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/instance_segmentation/maskrcnn_r50.yaml>`_                                       | MaskRCNN-ResNet50          | 533.80              | 177.90          |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------+---------------------+-----------------+
| `Instance Segmentation MaskRCNN SwinT <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/instance_segmentation/maskrcnn_swint.yaml>`_                                        | MaskRCNN-SwinT             | 566                 | 191.46          |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------+---------------------+-----------------+
| `Instance Segmentation RTMDet-Inst Tiny <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/instance_segmentation/rtmdet_inst_tiny.yaml>`_                                    | RTMDet-Ins-tiny            | 52.86               | 22.8            |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------+---------------------+-----------------+

Above table can be found using the following command

.. code-block:: shell

    (otx) ...$ otx find --task INSTANCE_SEGMENTATION

MaskRCNN-SwinT leverages `Swin Transformer <https://arxiv.org/abs/2103.14030>`_ architecture as its backbone network for feature extraction. This choice, while yielding superior accuracy, comes with a longer training time and higher computational requirements.

In contrast, the MaskRCNN-ResNet50 model adopts the more conventional ResNet-50 backbone network, striking a balance between accuracy and computational efficiency.

Meanwhile, MaskRCNN-EfficientNetB2B employs `EfficientNet-B2 <https://arxiv.org/abs/1905.11946>`_ architecture as its backbone, offering a compromise between accuracy and speed during training, making it a favorable option when minimizing training time and computational resources is essential.

Recently, we have updated RTMDet-Ins-tiny, integrating works from `RTMNet <https://arxiv.org/abs/2212.07784>`_ to prioritize real-time instance segmentation inference. While this model is tailored for real-time applications due to its lightweight design, it may not achieve the same level of accuracy as its counterparts, potentially necessitating more extensive training data.

Our experiments indicate that MaskRCNN-SwinT and MaskRCNN-ResNet50 outperform MaskRCNN-EfficientNetB2B in terms of accuracy. However, if reducing training time is paramount, transitioning to MaskRCNN-EfficientNetB2B is recommended. Conversely, for applications where inference speed is crucial, RTMDet-Ins-tiny presents an optimal solution.

In the table below the `mAP <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ metric on some academic datasets using our :ref:`supervised pipeline <instance_segmentation_supervised_pipeline>` is presented. The results were obtained on our recipes without any changes. We use 1024x1024 image resolution, for other hyperparameters, please, refer to the related recipe. We trained each model with single Nvidia GeForce RTX3090.

+---------------------------+--------------+------------+-----------------+
| Model name                | ADE20k       | Cityscapes | Pascal-VOC 2007 |
+===========================+==============+============+=================+
| MaskRCNN-EfficientNetB2B  | N/A          | N/A        | N/A             |
+---------------------------+--------------+------------+-----------------+
| MaskRCNN-ResNet50         | N/A          | N/A        | N/A             |
+---------------------------+--------------+------------+-----------------+
| MaskRCNN-SwinT            | N/A          | N/A        | N/A             |
+---------------------------+--------------+------------+-----------------+
| RTMDet-Ins-tiny           | N/A          | N/A        | N/A             |
+---------------------------+--------------+------------+-----------------+