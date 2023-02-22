############################
Use Self-Supervised Learning
############################

This tutorial introduces how to train a model using self-supervised learning and fine-tune the model with pre-trained weights.
OTX provides self-supervised learning methods for :doc:`multi-classification <../../explanation/algorithms/classification/multi_class_classification>` and :doc:`semantic segmentation <../../explanation/algorithms/segmentation/semantic_segmentation>`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.7

.. note::

  This example demonstates how to work with :doc:`self-supervised learning for classification <../../explanation/algorithms/classification/multi_class_classification>` and :doc:`self-supervised learning for semantic segmentation <../../explanation/algorithms/segmentation/semantic_segmentation>`.

*************************
Setup virtual environment
*************************

To create a universal virtual environment for OTX, please follow the installation process in the :doc:`quick start guide <../../get_started/quick_start_guide/installation>`.

***************************
Multi-class classification
***************************

.. note::

  To prepare `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_, refer to :doc:`how to train classification models <../base/how_to_train/classification>`.

Pre-training
------------

1. We can check which models are provided for this task like the command below. All details are in :doc:`how to train classification models <../base/how_to_train/classification>`. We will choose :ref:`MobileNet-V3-large-1x <classificaiton_models>` like :doc:`how to train classification models <../base/how_to_train/classification>`.

.. code-block::

    (otx) ...$ otx find --task classification

    +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
    |      TASK      |                         ID                        |          NAME         |                                        PATH                                       |
    +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+
    | CLASSIFICATION | Custom_Image_Classification_MobileNet-V3-large-1x | MobileNet-V3-large-1x | otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml |
    | CLASSIFICATION |    Custom_Image_Classification_EfficinetNet-B0    |    EfficientNet-B0    |    otx/algorithms/classification/configs/efficientnet_b0_cls_incr/template.yaml   |
    | CLASSIFICATION |   Custom_Image_Classification_EfficientNet-V2-S   |   EfficientNet-V2-S   |   otx/algorithms/classification/configs/efficientnet_v2_s_cls_incr/template.yaml  |
    +----------------+---------------------------------------------------+-----------------------+-----------------------------------------------------------------------------------+

2. Prepare an OTX workspace by running the following command:

.. code-block::

    (otx) ...$ otx build --train-data-roots data/flower_photos --model MobileNet-V3-large-1x --train-type SELFSUPERVISED

    [*] Workspace Path: otx-workspace-CLASSIFICATION
    [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
    [*] Load Model Name: MobileNet-V3-large-1x
    [*]     - Updated: otx-workspace-CLASSIFICATION/selfsl/model.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/selfsl/data_pipeline.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/deployment.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/hpo_config.yaml
    [*]     - Updated: otx-workspace-CLASSIFICATION/model_hierarchical.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/model_multilabel.py
    [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml

    (otx) ...$ cd ./otx-workspace-CLASSIFICATION

Unlike :doc:`how to train classification models <../base/how_to_train/classification>`, ``--train-type=SELFSUPERVISED`` must be added to get training components for self-supervised learning.

3. To start training we need to call ``otx train`` command in our worspace.

.. note::
    It is recommended to set ``--save-model-to`` to distinguish between pre-trained and fine-tuned weights or not to overwrite them.

.. code-block::

  (otx) ...$ otx train --save-model-to=models/selfsl

The training will return artifacts: ``weights.pth`` and ``label_schema.json`` and we can use this weights to fine-tune the models using target dataset.


Fine-tuning
-----------

1. Update our workspace to enable supervised (incremental) learning, which we actually try to do.
Call the command below from `the root directory` without adding ``--train-type=SELFSUPERVISED`` in the command.

.. code-block::

    (otx) ...$ otx build --train-data-roots data/flower_photos --model MobileNet-V3-large-1x

    [*] Workspace Path: otx-workspace-CLASSIFICATION
    [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
    [*] Load Model Name: MobileNet-V3-large-1x
    [*]     - Updated: otx-workspace-CLASSIFICATION/model.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/data_pipeline.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/deployment.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/hpo_config.yaml
    [*]     - Updated: otx-workspace-CLASSIFICATION/model_hierarchical.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/model_multilabel.py
    [*]     - Updated: otx-workspace-CLASSIFICATION/compression_config.json
    [*] Found validation data in your dataset in /home/sungchul/workspace/src/training_extensions/dataset/flower_photos. It'll be used as validation data.
    [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml

    (otx) ...$ cd ./otx-workspace-CLASSIFICATION

1. To start training we need to call the below command with adding ``--load-weights`` argument in our worspace.

.. note::
    It is recommended to set ``--save-model-to`` to distinguish between pre-trained and fine-tuned weights or not to overwrite them.

.. code-block::

  (otx) ...$ otx train --load-weights=models/selfsl/weights.pth --save-model-to=models/finetune

After these progesses, you can validate, optimize, and export the models described in :doc:`how to train classification models <../base/how_to_train/classification>`.


***************************
Semantic segmentation
***************************