Multi-class Classification
==========================

Multi-class Classification is the problem of classifying instances into one of two or more classes. We solve this problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the distribution probability of the categories from the given corpus.
For the supervised training we use the following algorithms components:

.. _mcl_cls_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip and random rotate, we use `Augmix <https://arxiv.org/abs/1912.02781>`_. This advanced type of augmentations helps to significantly expand the training distribution.

- ``Optimizer``: `Sharpness Aware Minimization (SAM) <https://arxiv.org/abs/2209.06585>`_. Wrapper upon the `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer that helps to achieve better generalization minimizing simultaneously loss value and loss sharpness.

- ``Learning rate schedule``: `Cosine Annealing <https://arxiv.org/abs/1608.03983v5>`_. It is a common learning rate scheduler that tends to work well on average for this task on a variety of different datasets.

- ``Loss function``: `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. This is a balancing training method that helps us to solve the imbalanced data problem.

- Additionally, we use `No Bias Decay (NBD) <https://arxiv.org/abs/1812.01187>`_ technique and **early stopping** to add adaptability to the training pipeline and prevent overfitting.
To further enhance the performance of the algorithm in case when we have a small number of data we use `Supervised Contrastive Learning <https://arxiv.org/abs/2004.11362>`_. More specifically, we train a model with two heads: classification head with Influence-Balanced Loss and SupCon head with `Barlow Twins loss <https://arxiv.org/abs/2103.03230>`_.

**************
Dataset Format
**************

We support a commonly used format for multi-class image classification task: `imagenet <https://www.image-net.org/>`_ class folder format.
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

With dataset format you can simply run OTX training with one line:

.. code-block::

    $ otx {train, optimize} <model_template> --train-data-root <path_to_data_root>

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classificaiton models.

******
Models
******
.. _classificaiton_models:

We support the following ready-to-use model templates:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                                                                  | Name                   | Complexity (GFLOPs) | Model size (MB) |
+==============================================================================================================================================================================================================================+========================+=====================+=================+
| `Custom_Image_Classification_MobileNet-V3-large-1x <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml>`_             | MobileNet-V3-large-1x  | 0.44                | 4.29            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficinetNet-B0 <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml>`_                        | EfficientNet-B0        | 0.81                | 4.09            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficientNet-V2-S <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml>`_                    | EfficientNet-V2-S      | 5.76                | 20.23           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

`EfficientNet-V2-S <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance. `MobileNet-V3-large-1x <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this template provides competitive accuracy as well.
`EfficientNet-B0 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets, but may be not so stable in case of a small amount of training data.

Besides this, we support public backbones from `torchvision <https://pytorch.org/vision/stable/index.html>`_, `pytorchcv <https://github.com/osmr/imgclsmob>`_, `mmcls <https://github.com/open-mmlab/mmclassification>`_ and `OpenVino Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`_.
Please, refer to the :doc:`tutorial <../../tutorials/advanced/backbones.rst>`_ how to customize models and run public backbones.

To see which public backbones are available for the task, the following command can be executed:

.. code-block::

        $ otx find --backbone {torchvision, pytorchcv, mmcls, omz.mmcls}

In the table below the top-1 accuracy on some academic datasets using our :ref:`supervised pipeline <mcl_cls_supervised_pipeline>` is presented. The results were obtained on our templates without any changes. We use 224x224 image resolution, for other hyperparameters, please, refer to the related template. We trained all models on 1 GPU Nvidia GeForce GTX3090.

+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| Model name            | CIFAR100        |cars       |flowers    | pets      |SVHN       |
+=======================+=================+===========+===========+===========+===========+
| MobileNet-V3-large-1x | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| EfficientNet-B0       | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| EfficientNet-V2-S     | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+

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

To be added soonMulti-class Classification
==========================

Multi-class Classification is the problem of classifying instances into one of two or more classes. We solve this problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the distribution probability of the categories from the given corpus.
For the supervised training we use the following algorithms components:

.. _mcl_cls_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip and random rotate, we use `Augmix <https://arxiv.org/abs/1912.02781>`_. This advanced type of augmentations helps to significantly expand the training distribution.

- ``Optimizer``: `Sharpness Aware Minimization (SAM) <https://arxiv.org/abs/2209.06585>`_. Wrapper upon the `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer that helps to achieve better generalization minimizing simultaneously loss value and loss sharpness.

- ``Learning rate schedule``: `Cosine Annealing <https://arxiv.org/abs/1608.03983v5>`_. It is a common learning rate scheduler that tends to work well on average for this task on a variety of different datasets.

- ``Loss function``: `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. This is a balancing training method that helps us to solve the imbalanced data problem.

- Additionally, we use `No Bias Decay (NBD) <https://arxiv.org/abs/1812.01187>`_ technique and **early stopping** to add adaptability to the training pipeline and prevent overfitting.
To further enhance the performance of the algorithm in case when we have a small number of data we use `Supervised Contrastive Learning <https://arxiv.org/abs/2004.11362>`_. More specifically, we train a model with two heads: classification head with Influence-Balanced Loss and SupCon head with `Barlow Twins loss <https://arxiv.org/abs/2103.03230>`_.

**************
Dataset Format
**************

We support a commonly used format for multi-class image classification task: `imagenet <https://www.image-net.org/>`_ class folder format.
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

With dataset format you can simply run OTX training with one line:

.. code-block::

    $ otx {train, optimize} <model_template> --train-data-root <path_to_data_root>

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classificaiton models.

******
Models
******
.. _classificaiton_models:

We support the following ready-to-use model templates:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                                                                  | Name                   | Complexity (GFLOPs) | Model size (MB) |
+==============================================================================================================================================================================================================================+========================+=====================+=================+
| `Custom_Image_Classification_MobileNet-V3-large-1x <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml>`_             | MobileNet-V3-large-1x  | 0.44                | 4.29            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficinetNet-B0 <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml>`_                        | EfficientNet-B0        | 0.81                | 4.09            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficientNet-V2-S <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml>`_                    | EfficientNet-V2-S      | 5.76                | 20.23           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

`EfficientNet-V2-S <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance. `MobileNet-V3-large-1x <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this template provides competitive accuracy as well.
`EfficientNet-B0 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets, but may be not so stable in case of a small amount of training data.

Besides this, we support public backbones from `torchvision <https://pytorch.org/vision/stable/index.html>`_, `pytorchcv <https://github.com/osmr/imgclsmob>`_, `mmcls <https://github.com/open-mmlab/mmclassification>`_ and `OpenVino Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`_.
Please, refer to the :doc:`tutorial <../../tutorials/advanced/backbones.rst>`_ how to customize models and run public backbones.

To see which public backbones are available for the task, the following command can be executed:

.. code-block::

        $ otx find --backbone {torchvision, pytorchcv, mmcls, omz.mmcls}

In the table below the top-1 accuracy on some academic datasets using our :ref:`supervised pipeline <mcl_cls_supervised_pipeline>` is presented. The results were obtained on our templates without any changes. We use 224x224 image resolution, for other hyperparameters, please, refer to the related template. We trained all models on 1 GPU Nvidia GeForce GTX3090.

+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| Model name            | CIFAR100        |cars       |flowers    | pets      |SVHN       |
+=======================+=================+===========+===========+===========+===========+
| MobileNet-V3-large-1x | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| EfficientNet-B0       | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+
| EfficientNet-V2-S     | N/A             | N/A       | N/A       | N/A       | N/A       |
+-----------------------+-----------------+-----------+-----------+-----------+-----------+

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