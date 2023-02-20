Multi-class Classification
==========================

Multi-class classification is the problem of classifying instances into one of two or more classes. We solve this problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the distribution probability of the categories from the given corpus.
For the supervised training we use the following algorithms components:

.. _mcl_cls_supervised_pipeline:

- ``Augmentations``: Besides basic augmentations like random flip and random rotate, we use `Augmix <https://arxiv.org/abs/1912.02781>`_. This advanced type of augmentations helps to significantly expand the training distribution.

- ``Optimizer``: `Sharpness Aware Minimization (SAM) <https://arxiv.org/abs/2209.06585>`_. Wrapper upon the `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer that helps to achieve better generalization minimizing simultaneously loss value and loss sharpness.

- ``Learning rate schedule``: `Cosine Annealing <https://arxiv.org/abs/1608.03983v5>`_. It is a common learning rate scheduler that tends to work well on average for this task on a variety of different datasets.

- ``Loss function``: We use standart `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model. However, for the class-incremental scenario we use `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. IB loss is a solution for class-imbalance, which avoids overfitting to the majority classes re-weighting the influential samples.

- ``Training technique``
    - `No Bias Decay (NBD) <https://arxiv.org/abs/1812.01187>`_: To add adaptability to the training pipeline and prevent overfitting.
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting. You can use early stopping like the below command.
      
      .. code-block::

        $ otx train {TEMPLATE} ... \
            params \
            --learning_parameters.enable_early_stopping=True \      # is early stopping used
            --learning_parameters.early_stop_start=3 \              # the number of epochs (iters) in which early stopping proceeds
            --learning_parameters.early_stop_patience=8 \           # (for epoch runner) stop if the model don't improve within the number of epochs of patience
            --learning_parameters.early_stop_iteration_patience=8 \ # (for iter runner) stop if the model don't improve within the number of iterations of patience

    - `Balanced Sampler <https://github.dev/openvinotoolkit/training_extensions/blob/develop/otx/mpa/modules/datasets/samplers/balanced_sampler.py#L11>`_: To create an efficient batch that consists of balanced samples over classes, reducing the iteration size as well.
    - `Supervised Contrastive Learning (SupCon) <https://arxiv.org/abs/2004.11362>`_: To enhance the performance of the algorithm in case when we have a small number of data. More specifically, we train a model with two heads: classification head with Influence-Balanced Loss and contrastive head with `Barlow Twins loss <https://arxiv.org/abs/2103.03230>`_. It enables using `--learning_parameters.enable_supcon=True` in CLI.
      The below table shows how much performance SupCon improved compared with baseline performance on three baseline datasets with 10 samples per class: CIFAR10, Eurosat-10, and Food-101.

        +-----------------------+---------+------------+----------+
        | Model name            | CIFAR10 | Eurosat-10 | Food-101 |
        +=======================+=========+============+==========+
        | MobileNet-V3-large-1x | +3.82   | +1.10      | -0.35    |
        +-----------------------+---------+------------+----------+
        | EfficientNet-B0       | +3.54   | +3.36      | +1.91    |
        +-----------------------+---------+------------+----------+
        | EfficientNet-V2-S     | +3.35   | +1.28      | +3.52    |
        +-----------------------+---------+------------+----------+

      You can use SupCon training like the below command.

      .. code-block::

        $ otx train {TEMPLATE} ... \
            params \
            --learning_parameters.enable_supcon=True

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

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classificaiton models.

******
Models
******
.. _classificaiton_models:

We support the following ready-to-use model templates:

+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                                                  | Name                  | Complexity (GFLOPs) | Model size (MB) |
+==============================================================================================================================================================================================================+=======================+=====================+=================+
| `Custom_Image_Classification_MobileNet-V3-large-1x <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml>`_ | MobileNet-V3-large-1x | 0.44                | 4.29            |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficinetNet-B0 <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml>`_            | EfficientNet-B0       | 0.81                | 4.09            |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficientNet-V2-S <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml>`_        | EfficientNet-V2-S     | 5.76                | 20.23           |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------+---------------------+-----------------+

`EfficientNet-V2-S <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance. `MobileNet-V3-large-1x <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this template provides competitive accuracy as well.
`EfficientNet-B0 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets, but may be not so stable in case of a small amount of training data.

Besides this, we support public backbones from `torchvision <https://pytorch.org/vision/stable/index.html>`_, `pytorchcv <https://github.com/osmr/imgclsmob>`_, `mmcls <https://github.com/open-mmlab/mmclassification>`_ and `OpenVino Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`_.
Please, refer to the :doc:`tutorial <../../../tutorials/advanced/backbones>` how to customize models and run public backbones.

To see which public backbones are available for the task, the following command can be executed:

.. code-block::

        $ otx find --backbone {torchvision, pytorchcv, mmcls, omz.mmcls}

In the table below the top-1 accuracy on some academic datasets using our :ref:`supervised pipeline <mcl_cls_supervised_pipeline>` is presented. The results were obtained on our templates without any changes. We use 224x224 image resolution, for other hyperparameters, please, refer to the related template. We trained each model with single Nvidia GeForce RTX3090.

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

Self-supervised learning can be one of the solutions if the user has a small data set, but label information is not yet available.
General self-supervised Learning in academia is commonly used to obtain well-pretrained weights from a source dataset without label information.
However, in real-world industries, it is difficult to apply because of small datasets, limited resources, or training in minutes.

For these cases, OTX provides improved self-supervised learning recipes that can be applied to the above harsh environments.
We adapted `BYOL <https://arxiv.org/abs/2006.07733>`_ as our self-supervised method.
Users only need a few more minutes to use these self-supervised learning recipes and can expect improved performance, especially in low-data regimes.

Below is graphs of performance improvement for three baseline datasets: CIFAR10, CIFAR100, and Food-101.
The graphs below show how much performance improvement over baseline was achieved using our self-supervised learning recipes.
In particular, the smaller the data, the greater the performance improvement can be expected.

.. image:: ../../../../utils/images/multi_cls_selfsl_performance_CIFAR10.png
  :width: 600

.. image:: ../../../../utils/images/multi_cls_selfsl_performance_CIFAR100.png
  :width: 600

.. image:: ../../../../utils/images/multi_cls_selfsl_performance_Food-101.png
  :width: 600


********************
Incremental Learning
********************

To be added soon