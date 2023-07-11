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

As it is a common practice to use object detection datasets in the academic area, we support the most popular object detection format: `COCO <https://cocodataset.org/#format-data>`_.
Specifically, this format should be converted in our `internal representation <https://github.com/openvinotoolkit/training_extensions/tree/develop/data/datumaro_multilabel>`_ first. We provided a `script <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/classification/utils/convert_coco_to_multilabel.py>` to help with conversion.
To convert the COCO data format to our internal one, run this script in similar way:

.. code-block::
    python convert_coco_to_multilabel.py --ann_file_path <path to .json COCO annotations> --data_root_dir <path to images folder> --output <output path to save annotations>

.. note::
    Names of the annotations files and overall dataset structure should be the same as the original `COCO <https://cocodataset.org/#format-data>`_. You need to convert train and validation sets separately.

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classification models.

******
Models
******
We use the same models as for Multi-class classification. Please, refer: :ref:`Classification Models <classification_models>`.

In the table below the `mAP <https://en.wikipedia.org/w/index.php?title=Information_retrieval&oldid=793358396#Average_precision>`_ metrics on some academic datasets using our :ref:`supervised pipeline <ml_cls_supervised_pipeline>` are presented. The results were obtained on our templates without any changes (including input resolution, which is 224x224 for all templates). We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+-----------------+-----------+------------------+-----------+
| Model name            | Pascal-VOC 2007 | COCO 2014 | Aerial Maritime  | Mean mAP  |
+=======================+=================+===========+==================+===========+
| MobileNet-V3-large-1x | 86.14           | 67.94     | 69.61            | 74.56     |
+-----------------------+-----------------+-----------+------------------+-----------+
| EfficientNet-B0       | 86.07           | 67.87     | 73.83            | 75.92     |
+-----------------------+-----------------+-----------+------------------+-----------+
| EfficientNet-V2-S     | 91.91           | 77.28     | 71.52            | 80.24     |
+-----------------------+-----------------+-----------+------------------+-----------+

************************
Semi-supervised Learning
************************

Semi-SL (Semi-supervised Learning) is a type of machine learning algorithm that uses both labeled and unlabeled data to improve the performance of the model. This is particularly useful when labeled data is limited, expensive or time-consuming to obtain.

To utilize unlabeled data during training, we use `BarlowTwins loss <https://arxiv.org/abs/2103.03230>`_ as an auxiliary loss for Semi-SL task solving. BarlowTwins enforces consistency across augmented versions of the same data (both labeled and unlabeled): each sample is augmented first with `Augmix <https://arxiv.org/abs/1912.02781>`_, then strongly augmented sample is generated by applying a pre-defined `RandAugment <https://arxiv.org/abs/1909.13719>`_ strategy on top of the basic augmentation.

.. _mlc_cls_semi_supervised_pipeline:

- ``BarlowTwins loss``: A specific implementation of Semi-SL that combines the use of a consistency loss with strong data augmentations, and a specific optimizer called Sharpness-Aware Minimization (`SAM <https://arxiv.org/abs/2010.01412>`_) to improve the performance of the model.

- ``Adaptive loss auxiliary loss weighting``: A technique for assigning such a weight for an auxiliary loss that the resulting value is a predefined fraction of the EMA-smoothed main loss value. This method allows aligning contribution of the losses during different training phases.

- ``Exponential Moving Average (EMA)``: A technique for maintaining a moving average of the model's parameters, which can improve the generalization performance of the model.

- ``Additional techniques``: Other than that, we use several solutions that apply to supervised learning (No bias Decay, Augmentations, Early-Stopping, etc.)

Please, refer to the :doc:`tutorial <../../../tutorials/advanced/semi_sl>` on how to train semi-supervised learning.
Training time depends on the number of images and can be up to several times longer than conventional supervised learning.

In the table below the mAP metric on some public datasets using our pipeline is presented.

+-----------------------+---------+----------------------+----------------+---------+----------------+---------+
|        Dataset        | AerialMaritime 3 cls |         | VOC 2007 3 cls |         | COCO 14 5 cls  |         |
+=======================+======================+=========+================+=========+================+=========+
|                       |   SL                 | Semi-SL |  SL            | Semi-SL |   SL           | Semi-SL |
+-----------------------+----------------------+---------+----------------+---------+----------------+---------+
| MobileNet-V3-large-1x |  74.28               |  74.41  | 96.34          |  97.29  |  82.39         |  83.77  |
+-----------------------+----------------------+---------+----------------+---------+----------------+---------+
|   EfficientNet-B0     |  79.59               |  80.91  | 97.75          |  98.59  | 83.24          |  84.19  |
+-----------------------+----------------------+---------+----------------+---------+----------------+---------+
|  EfficientNet-V2-S    |  75.91               |  81.91  | 95.65          |  96.43  | 85.19          |  84.24  |
+-----------------------+----------------------+---------+----------------+---------+----------------+---------+

AerialMaritime was sampled with 5 images per class. VOC was sampled with 10 images per class and COCO was sampled with 20 images per class.
Additionel information abount the datasets can be found in the table below.

+-----------------------+----------------+----------------------+
|        Dataset        | Labeled images | Unlabeled images     |
+=======================+================+======================+
| AerialMaritime 3 cls  |  10            |  42                  |
+-----------------------+----------------+----------------------+
| VOC 2007 3 cls        |  30            |  798                 |
+-----------------------+----------------+----------------------+
| COCO 14 5 cls         |  95            |  10142               |
+-----------------------+----------------+----------------------+

.. note::
    This result can vary depending on the image selected for each class. Also, since there are few labeled settings for the Semi-SL algorithm. Some models may require larger datasets for better results.

.. ************************
.. Self-supervised Learning
.. ************************

.. To be added soon

.. ********************
.. Incremental Learning
.. ********************

.. To be added soon