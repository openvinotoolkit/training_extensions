Multi-class Classification
==========================

**************
Dataset format
**************

We support a commonly used format for classification tasks: `imagenet <https://www.image-net.org/>`_ class folder format.
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

*********
Backbones
*********

We support the following ready-to-use templates:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| Template ID                                                                                                                                                                                                                  | Name                   | Complexity (GFlops) | Model size (MB) |
+==============================================================================================================================================================================================================================+========================+=====================+=================+
| `Custom_Image_Classification_MobileNet-V3-large-1x <https://github.com/openvinotoolkit/training_extensions/tree/0d98bcd21d5e441516b8ec06949bc84870102b3f/otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr>`_ | MobileNet-V3-large-1x  | 0.44                | 4.29            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficinetNet-B0 <https://github.com/openvinotoolkit/training_extensions/tree/0d98bcd21d5e441516b8ec06949bc84870102b3f/otx/algorithms/classification/configs/efficientnet_b0_cls_incr>`_         | EfficientNet-B0        | 0.81                | 4.09            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
| `Custom_Image_Classification_EfficientNet-V2-S <https://github.com/openvinotoolkit/training_extensions/tree/0d98bcd21d5e441516b8ec06949bc84870102b3f/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr>`_  | EfficientNet-V2-S      | 5.76                | 20.23           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

`EfficientNet-V2-S <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance. `MobileNet-V3-large-1x <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this template provides competitive accuracy as well.
`EfficientNet-B0 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets, but may be not so stable in case of a small amount of training data.

Besides this, we support public backbones from `torchvision <https://pytorch.org/vision/stable/index.html>`_, `pytorchcv <https://github.com/osmr/imgclsmob>`_ and `OpenVino Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`_.
Please, refer to `tutorial <N/A>`_ how to run public backbones.

To see which public backbones are available for the task, the following command can be executed:
.. codeblock::

   $ otx find --backbone {torchvision, pytorchcv, mmcls, omz.mmcls}


*******************************
Supervised Incremental Learning
*******************************

We solve the multi-class classification problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the probability of each label presented on the image.
For the supervised training we use the following algorithms components:

- ``Augmentations``: Besides basic augmentations like random flip and random rotate, we use `Augmix <https://arxiv.org/abs/1912.02781>`_. This advanced type of augmentation helps to significantly expand the training distribution.

- ``Optimizer``: `Sharpness Aware Minimization (SAM) <https://arxiv.org/abs/2209.06585>`_. Wrapper upon the SGD that helps to achieve better generalization minimizing simultaneously loss value and loss sharpness.

- ``Learning rate schedule``: `Cosine Annealing <https://arxiv.org/abs/1608.03983v5>`_. It is a common learning rate scheduler that tends to work well on average for this task on variety of different datasets.

- ``Loss function``: `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. This is a balancing training method that helps us to solve the imbalanced data problem.

- Additionaly, we use `No Bias Decay (NBD) <https://arxiv.org/abs/1812.01187>`_ technique and **early stopping** to add adaptability to the training pipeline and prevent overfitting.
To further enhance the performance of the algorithm in case when we have a small number of data we use `Supervised Contrastive Learning <https://arxiv.org/abs/2004.11362>`_. More concretely, we train a model with two heads: classification head with Influence-Balanced Loss and SupCon head with `Barlow Twins loss <https://arxiv.org/abs/2103.03230>`_.

In the table below the top-1 accuracy on some academic datasets is presented. The results were obtained on our templates without any changes. We use 240x240 image resolution, for other hyperparameters, please, refer to the related template. We train all models on 1 GPU Nvidia GeForce GTX3090.

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