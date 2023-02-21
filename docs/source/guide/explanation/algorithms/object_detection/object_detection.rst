Object Detection
================

**************
Dataset Format
**************

To be added soon

******
Models
******

To be added soon

************************
Semi-supervised Learning
************************

For Semi-SL task solving we use `Unbiased Teacher model <https://arxiv.org/abs/2102.09480>`_, which is a specific implementation of Semi-SL for object detection. Unbiased teacher detach the student model and the teacher model to prevent teacher from being polluted by noisy pseudo-labels. In the early stage, the teacher model is trained by supervised loss. This stage is called a burn-in stage. After the burn-in, the student model is trained using both pseudo-labeled data from the teacher model and labeled data. And the teacher model is updated using
EMA.

In Semi-SL, the pseudo-labeling process is combined with a consistency loss that ensures that the predictions of the model are consistent across augmented versions of the same data. This helps to reduce the impact of noisy or incorrect labels that may arise from the pseudo-labeling process. Additionally, our algorithm uses a combination of strong data augmentations and a specific optimizer called Sharpness-Aware Minimization (SAM) to further improve the accuracy of the model.

Overall, OTX utilizes powerful techniques for improving the performance of Semi-SL algorithm with limited labeled data. They can be particularly useful in domains where labeled data is expensive or difficult to obtain, and can help to reduce the time and cost associated with collecting labeled data.

.. _od_semi_supervised_pipeline:

- ``Pseudo-labeling``: A specific implementation of Semi-SL that combines the use of pseudo-labeling with a consistency loss, strong data augmentations, and a specific optimizer called Sharpness-Aware Minimization (SAM) to improve the performance of the model.

- ``Weak & Strong augmentation``: For teacher model weak augmentations(random flip) are applied to input image. For student model strong augmentations(colorjtter, grayscale, goussian blur, random erasing) are applied.

- ``Additional training techniques``: Other than that, we use several solutions that apply to supervised learning (No bias Decay, Augmentations, Early stopping, LR conditioning.).

Please, refer to the :doc:`tutorial <../../../tutorials/advanced/semi_sl>` how to train semi supervised learning. 

In the table below the mAP on toy data sample from `COCO <https://cocodataset.org/#home>`_ dataset using our pipeline is presented. 

We sample 400 images that contains one of [person, car, bus] for labeled train images. And 4000 images for unlabeled images. For validation 100 images are selected from val2017

+---------+--------------------------------------------+
| Dataset |            Sampled COCO dataset            |   
+=========+=====================+======================+
|         |          SL         |       Semi-SL        |
+---------+---------------------+----------------------+
|  ATSS   |  | Person: 69.70    | | Person: 69.44      |
|         |  | Car:    65.00    | | Car:    65.84      |
|         |  | Bus:    42.96    | | Bus:    50.7       |
|         |  | Mean:   59.20    | | Mean:   61.98      |
+---------+---------------------+----------------------+
|   SSD   | | Person: 39.24     | | Person: 38.52      |
|         | | Car:    19.24     | | Car:    28.02      |
|         | | Bus:    21.34     | | Bus:    26.28      |
|         | | Mean:   26.60     | | Mean:   30.96      |
+---------+---------------------+----------------------+
|  YOLOX  | | Person: 65.64     | | Person: 69.00      |
|         | | Car:    64.44     | | Car:   65.66       |
|         | | Bus:    60.68     | | Bus:   65.12       |
|         | | Mean:   63.6      | | Mean:  66.58       |
+---------+---------------------+----------------------+

************************
Self-supervised Learning
************************

To be added soon

********************
Incremental Learning
********************

To be added soon
