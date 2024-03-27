############################
Use Semi-Supervised Learning
############################

This tutorial provides an example of how to use semi-supervised learning with OpenVINO™ Training Extensions on the specific dataset.

OpenVINO™ Training Extensions now offers semi-supervised learning, which combines labeled and unlabeled data during training to improve model accuracy in case when we have a small amount of annotated data. Currently, this type of training is available for multi-class classification, object detection, and semantic segmentation.

Semi-supervised learning will soon be available for multi-label classification and instance segmentation as well.

If you want to learn more about the algorithms used in semi-supervised learning, please refer to the explanation section below:

- `Multi-class Classification <../../explanation/algorithms/classification/multi_class_classification.html#semi-supervised-learning>`__
- `Object Detection <../../explanation/algorithms/object_detection/object_detection.html#semi-supervised-learning>`__
- `Semantic Segmentation <../../explanation/algorithms/segmentation/semantic_segmentation.html#semi-supervised-learning>`__

In this tutorial, we use the MobileNet-V3-large-1x model for multi-class classification to cite an example of semi-supervised learning.

The process has been tested on the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.1


.. note::

  To learn how to export the trained model, refer to `classification export <../base/how_to_train/classification.html#export>`__.

  To learn how to optimize the trained model (.xml) with OpenVINO™ PTQ, refer to `classification optimization <../base/how_to_train/classification.html#optimization>`__.

  Currently, OpenVINO™ NNCF optimization doesn't support a full Semi-SL training algorithm. The accuracy-aware optimization will be executed on labeled data only.
  So, the performance drop may be more noticeable than after ordinary supervised training.

  To learn how to deploy the trained model, refer to :doc:`deploy <../base/deploy>`.

  To learn how to run the demo and visualize results, refer to :doc:`demo <../base/demo>`.

This tutorial explains how to train a model in semi-supervised learning mode and how to evaluate the resulting model.

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

***************************
Dataset preparation
***************************

We use the same dataset, `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_, as we do in :doc:`classification tutorial <../base/how_to_train/classification>`.

Since it is assumed that we have additional unlabeled images,
we make a use of ``tests/assets/imagenet_dataset`` for this purpose as an example.


***************************
Enable via ``otx build``
***************************

1. To enable semi-supervsied learning via ``otx build``, we need to add arguments ``--unlabeled-data-roots``.
OpenVINO™ Training Extensions receives the root path where unlabeled images are by ``--unlabeled-data-roots``.

We should put the path where unlabeled data are contained. It is all we need to change to run semisupervised training. OpenVINO™ Training Extensions will recognize this training type automatically.

.. note::

  OpenVINO™ Training Extensions automatically searches for all image files with JPG, JPEG, and PNG formats in the root path specified using the ``--unlabeled-data-roots`` option, even if there are other file formats present.
  The image files which are located in sub-folders (if threy presented) will be also collected for building unlabeled dataset.

  In this tutorial, we make use of auto-split functionality for the multi-class classification, which makes train/validation splits for the given dataset.

  For the details about auto-split, please refer to :doc:`auto-configuration <../../explanation/additional_features/auto_configuration>`.

.. code-block::

  (otx) ...$ otx build --train-data-roots data/flower_photos --unlabeled-data-roots tests/assets/imagenet_dataset --model MobileNet-V3-large-1x


  [*] Workspace Path: otx-workspace-CLASSIFICATION
  [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
  [*] Load Model Name: MobileNet-V3-large-1x
  [*]     - Updated: otx-workspace-CLASSIFICATION/semisl/model.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/semisl/data_pipeline.py
  ...
  [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml


  (otx) ...$ cd ./otx-workspace-CLASSIFICATION


2. To start training we need to call ``otx train``
command in our workspace:

.. code-block::

  (otx) ...$ otx train

In the train log, you can check that the train type is set to **Semisupervised** and related configurations are properly loaded as following:

.. code-block::

  ...
  2023-02-22 06:21:54,492 | INFO : called _init_recipe()
  2023-02-22 06:21:54,492 | INFO : train type = Semisupervised
  2023-02-22 06:21:54,492 | INFO : train type = Semisupervised - loading training_extensions/src/otx/recipes/stages/classification/semisl.yaml
  2023-02-22 06:21:54,500 | INFO : Replacing runner from EpochRunnerWithCancel to EpochRunnerWithCancel.
  2023-02-22 06:21:54,503 | INFO : initialized recipe = training_extensions/src/otx/recipes/stages/classification/semisl.yaml
  ...


After training ends, a trained model is saved in the ``models`` sub-directory in the workspace ``otx-workspace-CLASSIFICATION``.


***************************
Enable via ``otx train``
***************************

1. To enable semi-supervised learning directly via ``otx train``, we also need to add the argument ``--unlabeled-data-roots``
specifying a path to unlabeled images.

.. code-block::

  (otx) ...$ otx train src/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                      --train-data-roots data/flower_photos \
                      --unlabeled-data-roots tests/assets/imagenet_dataset

In the train log, you can check that the train type is set to **Semisupervised** and related configurations are properly loaded as following:

.. code-block::

  ...
  2023-02-22 06:21:54,492 | INFO : called _init_recipe()
  2023-02-22 06:21:54,492 | INFO : train type = Semisupervised
  2023-02-22 06:21:54,492 | INFO : train type = Semisupervised - loading training_extensions/src/otx/recipes/stages/classification/semisl.yaml
  2023-02-22 06:21:54,500 | INFO : Replacing runner from EpochRunnerWithCancel to EpochRunnerWithCancel.
  2023-02-22 06:21:54,503 | INFO : initialized recipe = training_extensions/src/otx/recipes/stages/classification/semisl.yaml
  ...


After training ends, a trained model is saved in the ``latest_trained_model`` sub-directory in the workspace named ``otx-workspace-CLASSIFICATION`` by default.


***************************
Validation
***************************

In the same manner with `the normal validation <../base/how_to_train/classification.html#validation>`__,
we can evaluate the trained model with auto-splitted validation dataset in the workspace and
save results to ``outputs/performance.json`` by the following command:


.. code-block::

  (otx) ...$ otx eval src/otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                      --test-data-roots splitted_dataset/val \
                      --load-weights models/weights.pth \
                      --output outputs
