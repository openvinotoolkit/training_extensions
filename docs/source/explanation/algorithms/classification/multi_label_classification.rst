Multi-label Classification
==========================

Multi-label classification is a generalization of multiclass classification. The main goal of the task is to predict a set of labels per image. Each image may belong to more than one class and may belong to none of them at all.

We solve this problem by optimizing small binary classification sub-tasks aimed to predict whether or not the specific category from the corpus is presented on the given image.

.. _ml_cls_supervised_pipeline:

For supervised learning we use the following algorithms components:

- ``Augmentations``: Besides basic augmentations like random flip and random rotate, we use `Augmix <https://arxiv.org/abs/1912.02781>`_. This advanced type of augmentation helps to significantly expand the training distribution.

- ``Optimizer``: `Sharpness Aware Minimization (SAM) <https://arxiv.org/abs/2209.06585>`_. Wrapper upon the `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer that helps to achieve better generalization minimizing simultaneously loss value and loss sharpness.

- ``Learning rate schedule``: `One Cycle Learning Rate policy <https://arxiv.org/abs/1708.07120>`_. It is the combination of gradually increasing the learning rate and gradually decreasing the momentum during the first half of the cycle, then gradually decreasing the learning rate and increasing the momentum during the latter half of the cycle.

- ``Loss function``: We use **Asymmetric Angular Margin Loss**. We can formulate this loss as follows: :math:`L_j (cos\Theta_j,y) = \frac{k}{s}y p_-^{\gamma^-}\log{p_+} + \frac{1-k}{s}(1-y)p_+^{\gamma^+}\log{p_-}`, where :math:`s` is a scale parameter, :math:`m` is an angular margin, :math:`k` is negative-positive weighting coefficient, :math:`\gamma^+` and :math:`\gamma^-` are weighting parameters. For further information about loss function, ablation studies, and experiments, please refer to our dedicated `paper <https://arxiv.org/abs/2209.06585>`_.

- Additionally, we use the `No Bias Decay (NBD) <https://arxiv.org/abs/1812.01187>`_ technique, **Exponential Moving Average (EMA)** for the model's weights and adaptive **early stopping** to add adaptability and prevent overfitting.

**************
Dataset Format
**************

As it is a common practice to use object detection datasets in the academic area, we support the most popular object detection formats: `COCO <https://cocodataset.org/#format-data>`_.
Specifically, these formats will be converted in our `internal representation <https://github.com/openvinotoolkit/training_extensions/tree/develop/data/datumaro_multilabel>`_ via the `Datumaro <https://github.com/openvinotoolkit/datumaro>`_ dataset handler.

.. note::
    Names of the annotations files and overall dataset structure should be the same as the original `COCO <https://cocodataset.org/#format-data>`_.

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classificaiton models.

******
Models
******
We use the same models as for Multi-class classification. Please, refer: :ref:`Classificaiton Models <classificaiton_models>`.

In the table below the `mAP <https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision>`_ metrics on some academic datasets using our :ref:`supervised pipeline <ml_cls_supervised_pipeline>` are presented. The results were obtained on our templates without any changes. We use 448x448 image resolution to make the results comparable with academic papers, for other hyperparameters, please, refer to the related template. We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+-----------------+-----------+-----------+-----------+
| Model name            | Pascal-VOC 2007 |    COCO   | NUS-WIDE  | Mean mAP  |
+=======================+=================+===========+===========+===========+
| MobileNet-V3-large-1x | N/A             | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+
| EfficientNet-B0       | N/A             | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+
| EfficientNet-V2-S     | N/A             | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+

************************
Semi-supervised Learning
************************

To be added soon

************************
Self-supervised Learning
************************

To be added soon

********************
Incremental Learning
********************

To be added soon