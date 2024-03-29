Object Detection
================

Object detection is a computer vision task where it's needed to locate objects, finding their bounding boxes coordinates together with defining class.
The input is an image, and the output is a pair of coordinates for bouding box corners and a class number for each detected object.

The common approach to building object detection architecture is to take a feature extractor (backbone), that can be inherited from the classification task.
Then goes a head that calculates coordinates and class probabilities based on aggregated information from the image.
Additionally, some architectures use `Feature Pyramid Network (FPN) <https://arxiv.org/abs/1612.03144>`_ to transfer and process feature maps from backbone to head and called neck.

For the supervised training we use the following algorithms components:

.. _od_supervised_pipeline:

- ``Augmentations``: We use random crop and random rotate, simple bright and color distortions and multiscale training for the training pipeline.

- ``Optimizer``: We use `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer with the weight decay set to **1e-4** and momentum set to **0.9**.

- ``Learning rate schedule``: `ReduceLROnPlateau <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>`_. This learning rate scheduler proved its efficiency in dataset-agnostic trainings, its logic is to drop LR after some time without improving the target accuracy metric. Also, we update it with ``iteration_patience`` parameter that ensures that a certain number of training iterations (steps through the dataset) were passed before dropping LR.

- ``Loss function``: We use `Generalized IoU Loss <https://giou.stanford.edu/>`_  for localization loss to train the ability of the model to find the coordinates of the objects. For the classification head, we use a standard `FocalLoss <https://arxiv.org/abs/1708.02002>`_.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting.
    - `Anchor clustering for SSD <https://arxiv.org/abs/2211.17170>`_: This model highly relies on predefined anchor boxes hyperparameter that impacts the size of objects, which can be detected. So before training, we collect object statistics within dataset, cluster them and modify anchor boxes sizes to fit the most for objects the model is going to detect.
    - ``Backbone pretraining``: we pretrained MobileNetV2 backbone on large `ImageNet21k <https://github.com/Alibaba-MIIL/ImageNet21K>`_ dataset to improve feature extractor and learn better and faster.


**************
Dataset Format
**************

At the current point we support `COCO <https://cocodataset.org/#format-data>`_ and
`Pascal-VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_ dataset formats.
Learn more about the formats by following the links above. Here is an example of expected format for COCO dataset:

.. code::

  ├── annotations/
      ├── instances_train.json
      ├── instances_val.json
      └── instances_test.json
  ├──images/
      (Split is optional)
      ├── train
      ├── val
      └── test

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/detection>` for more information how to train, validate and optimize detection models.

******
Models
******

We support the following ready-to-use model recipes:

+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| Recipe ID                                                                                                                                                  | Name                | Complexity (GFLOPs) | Model size (MB) |
+============================================================================================================================================================+=====================+=====================+=================+
| `Custom_Object_Detection_YOLOX <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/yolox_tiny.yaml>`_            |      YOLOX-TINY     | 6.5                 | 20.4            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Object_Detection_YOLOX_S <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/yolox_s.yaml>`_                    |       YOLOX_S       | 33.51               | 46.0            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Object_Detection_YOLOX_L <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/yolox_l.yaml>`_                    |       YOLOX_L       | 194.57              | 207.0           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Object_Detection_YOLOX_X <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/yolox_x.yaml>`_                    |       YOLOX_X       | 352.42              | 378.0           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Custom_Object_Detection_Gen3_SSD <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/ssd_mobilenetv2.yaml>`_    |         SSD         | 9.4                 | 7.6             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Custom_Object_Detection_Gen3_ATSS <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/atss_mobilenetv2.yaml>`_  |  MobileNetV2-ATSS   | 20.6                | 9.1             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `Object_Detection_ResNeXt101_ATSS <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/detection/atss_resnext101.yaml>`_    |   ResNeXt101-ATSS   | 434.75              | 344.0           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+

Above table can be found using the following command

.. code-block:: shell

    (otx) ...$ otx find --task DETECTION

`MobileNetV2-ATSS <https://arxiv.org/abs/1912.02424>`_ is a good medium-range model that works well and fast in most cases.
`SSD <https://arxiv.org/abs/1512.02325>`_ and `YOLOX <https://arxiv.org/abs/2107.08430>`_ are light models, that a perfect for the fastest inference on low-power hardware.
YOLOX achieved the same accuracy as SSD, and even outperforms its inference on CPU 1.5 times, but requires 3 times more time for training due to `Mosaic augmentation <https://arxiv.org/pdf/2004.10934.pdf>`_, which is even more than for ATSS.
So if you have resources for a long training, you can pick the YOLOX model.
ATSS still shows good performance among `RetinaNet <https://arxiv.org/abs/1708.02002>`_ based models. Therfore, We added ATSS with large scale backbone, ResNeXt101-ATSS. We integrated large ResNeXt101 backbone to our Custom ATSS head, and it shows good transfer learning performance.
In addition, we added a YOLOX variants to support users' diverse situations.

In the table below the test mAP on some academic datasets using our :ref:`supervised pipeline <od_supervised_pipeline>` is presented.

For `COCO <https://cocodataset.org/#home>`__ dataset the accuracy of pretrained weights is shown, and we report official COCO mAP with AP50.
Except for COCO, we report AP50 as performance metric.

5 datasets were selected as transfer learning datasets.
`BDD100K <https://www.bdd100k.com/>`_ is the largest dataset among we used. 70000 images are used as train images and 10000 images are used for validation.
`Brackish <https://public.roboflow.com/object-detection/brackish-underwater>`_ and `Plantdoc <https://public.roboflow.com/object-detection/plantdoc>`_ are datasets of medium size. They have around 10000 images for train and 1500 images for validation.
`BCCD <https://public.roboflow.com/object-detection/bccd>`_ and `Chess pieces <https://public.roboflow.com/object-detection/chess-full>`_ are datasets of small size. They have around 300 images for train and 100 images for validation.
We used our own recipes without any modification.
For hyperparameters, please, refer to the related recipe.
We trained each model with a single Nvidia GeForce RTX3090.

+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| Model name                 | COCO(AP50)       | BDD100K   | Brackish  | Plantdoc  | BCCD      | Chess pieces |
+============================+==================+===========+===========+===========+===========+==============+
| YOLOX-TINY                 | 31.0 (48.2)      | 24.8      | 96.3      | 51.5      | 88.5      | 99.2         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| SSD                        | 13.5             | 28.2      | 96.5      | 52.9      | 91.1      | 99.1         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| MobileNetV2-ATSS           | 32.5 (49.5)      | 40.2      | 99.1      | 63.4      | 93.4      | 99.1         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| ResNeXt101-ATSS            | 45.1 (63.8)      | 45.5      | 99.3      | 69.3      | 93.1      | 99.1         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| YOLOX-S                    | 40.3 (59.1)      | 37.1      | 93.6      | 54.8      | 92.7      | 98.8         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| YOLOX-L                    | 49.4 (67.1)      | 44.5      | 94.6      | 55.8      | 91.8      | 99.0         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
| YOLOX-X                    | 50.9 (68.4)      | 44.2      | 96.3      | 56.2      | 91.5      | 98.9         |
+----------------------------+------------------+-----------+-----------+-----------+-----------+--------------+
