Multi-class Classification
==========================

Multi-class classification is the problem of classifying instances into one of two or more classes. We solve this problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the distribution probability of the categories from the given corpus.
For the supervised training we use the following algorithms components:

.. _mcl_cls_supervised_pipeline:

- ``Learning rate schedule``: `ReduceLROnPlateau <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>`_. It is a common learning rate scheduler that tends to work well on average for this task on a variety of different datasets.

- ``Loss function``: We use standard `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model. However, for the class-incremental scenario we use `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. IB loss is a solution for the class imbalance, which avoids overfitting to the majority classes re-weighting the influential samples.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting.
    - `Balanced Sampler <https://github.dev/openvinotoolkit/training_extensions/blob/develop/src/otx/algo/samplers/balanced_sampler.py#L11>`_: To create an efficient batch that consists of balanced samples over classes, reducing the iteration size as well.

**************
Dataset Format
**************

We support a commonly used format for multi-class image classification task: `ImageNet <https://www.image-net.org/>`_ class folder format.
This format has the following structure:

::

    data
    ├── train
        ├── class 0
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
        ├── class 1
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
        ...
        └── class N
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
    └── val
        ...

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classification models.

******
Models
******
.. _classification_models:

We support the following ready-to-use model recipes:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| Recipe ID                                                                                                                                                                                                        | Name                  | Complexity (GFLOPs) | Model size (MB) |
+==================================================================================================================================================================================================================+=======================+=====================+=================+
| `Custom_Image_Classification_MobileNet-V3-large-1x <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/classification/multi_class_cls/otx_mobilenet_v3_large.yaml>`_             | MobileNet-V3-large-1x | 0.44                | 4.29            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficinetNet-B0 <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/classification/multi_class_cls/otx_efficientnet_b0.yaml>`_                      | EfficientNet-B0       | 0.81                | 4.09            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficientNet-V2-S <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/classification/multi_class_cls/otx_efficientnet_v2.yaml>`_                    | EfficientNet-V2-S     | 5.76                | 20.23           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+

`EfficientNet-V2-S <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance. `MobileNet-V3-large-1x <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this recipe provides competitive accuracy as well.
`EfficientNet-B0 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets, but may be not so stable in case of a small amount of training data.

To see which models are available for the task, the following command can be executed:

.. code-block:: shell

        (otx) ...$ otx find --task MULTI_CLASS_CLS

In the table below the top-1 accuracy on some academic datasets using our :ref:`supervised pipeline <mcl_cls_supervised_pipeline>` is presented. The results were obtained on our Recipes without any changes. We use 224x224 image resolution, for other hyperparameters, please, refer to the related recipe. We trained each model with single Nvidia GeForce RTX3090.

+-----------------------+-----------------+-----------+-----------+-----------+
| Model name            | CIFAR10         |CIFAR100   |flowers*   | cars*     |
+=======================+=================+===========+===========+===========+
| MobileNet-V3-large-1x | 93.36           | 83.01     | 96.45     | 83.24     |
+-----------------------+-----------------+-----------+-----------+-----------+
| EfficientNet-B0       | 94.86           | 84.73     | 96.86     | 85.70     |
+-----------------------+-----------------+-----------+-----------+-----------+
| EfficientNet-V2-S     | 96.13           | 90.36     | 97.68     | 86.74     |
+-----------------------+-----------------+-----------+-----------+-----------+

\* These datasets were splitted with auto-split (80% train, 20% test).
