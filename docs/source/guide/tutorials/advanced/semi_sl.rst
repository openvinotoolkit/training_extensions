############################
Use Semi-Supervised Learning
############################

This tutorial provides instructions on how to use semi-supervised learning with OTX.

OTX now offers semi-supervised learning, which combines labeled and unlabeled data during training to improve model accuracy in case when we have small amount of annotated data. Currently, this type of training is available for multi-class classification, object detection, and semantic segmentation.

Semi-supervised learning will soon be available for multi-label classification and instance segmentation as well.

If want to learn more about the algorithms used in semi-supervised learning, please refer to the explanation section below.

- :doc:`Multi-class Classification <../../explanation/algorithms/classification/multi_class_classification>`
- :doc:`Object Detection <../../explanation/algorithms/object_detection/object_detection>`
- :doc:`Semantic Segmentation <../../explanation/algorithms/segmentation/semantic_segmentation>`

In this tutorial, we use MobileNet-V3-large-1x model in multi-class classification to cite an example for semi-supervised learning.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.1


.. note::
  
  Currently, semi-supervised learning models trained in OTX cannot be exported or optimized. However, we will support for this functionality in the near future.

  This tutorial explains how to train a model in semi-supervised learning mode and how to evaluate the resulting model.


***************************
Dataset preparation
***************************

We use the same dataset, `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_, as we do in :doc:`Classification Tutorial <../base/how_to_train/classification>`.

Since it is assumed that we have a bunch of unlabeled data, which is not annotated but only image files exist in semi-supervised learning,
we make a use of ``tests/assets/imagenet_dataset`` temporarily.


***************************
Enable via ``otx build``
***************************

1. To enable semi-supervsied learning via ``otx build``, we need to add arguments ``--unlabeled-data-roots`` and ``--train-type``. OTX receives the root path where unlabeled images are by ``--unlabeled-data-roots``.
We should put the path where unlabeled data are contained. OTX also provides us ``--train-type`` to select the type of training scheme. All we have to do for that is specifying it as **SEMISUPERVISED**.

.. note::

  OTX automatically search for all image files with JPG, JPEG, and PNG formats in the root path specified using the ``--unlabeled-data-roots`` option, even if there are other file formats present. The image files which are located in sub-folders will be also collected for building unlabeled dataset.

.. code-block::

  (otx) ...$ otx build --train-data-roots data/flower_photos --unlabeled-data-roots tests/assets/imagenet_dataset --model MobileNet-V3-large-1x --train-type SEMISUPERVISED
  

  [*] Workspace Path: otx-workspace-CLASSIFICATION
  [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
  [*] Load Model Name: MobileNet-V3-large-1x
  [*]     - Updated: otx-workspace-CLASSIFICATION/semisl/model.py
  [*]     - Updated: otx-workspace-CLASSIFICATION/semisl/data_pipeline.py
  ...
  [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml
  
  
  (otx) ...$ cd ./otx-workspace-CLASSIFICATION


2. To start training we need to call ``otx train``
command in our worspace:

.. code-block::

  (otx) ...$ otx train

In the train log, you can check that the train type is set to **SEMISUPERVISED** and related configurations are properly loaded as following:

.. code-block::

  ...
  2023-02-22 06:21:54,492 | INFO : called _init_recipe()
  2023-02-22 06:21:54,492 | INFO : train type = SEMISUPERVISED
  2023-02-22 06:21:54,492 | INFO : train type = SEMISUPERVISED - loading training_extensions/otx/recipes/stages/classification/semisl.yaml
  2023-02-22 06:21:54,500 | INFO : Replacing runner from EpochRunnerWithCancel to EpochRunnerWithCancel.
  2023-02-22 06:21:54,503 | INFO : initialized recipe = training_extensions/otx/recipes/stages/classification/semisl.yaml
  ...


After training ends, a trained model is saved in ``models`` sub-directory in workspace ``otx-workspace-CLASSIFICATION``.


3. In the same manner with :doc:`the normal validation <../base/how_to_train/classification#validation>`, 
we can evaluate the trained model in ``models`` folder with auto-splitted validation dataset 
and save results to ``performance.json`` by the following command:

.. code-block::

  (otx) ...$ otx eval --test-data-roots splitted_dataset/val \
                      --load-weights models/weights.pth \
                      --save-performance performance.json

***************************
Enable via ``otx train``
***************************

1. To enable semi-supervised learning directly via ``otx train``, we need to add arguments ``--unlabeled-data-roots`` and ``--algo_backend.train_type`` 
which is one of template-specific parameters (The details are provided in :doc:`quick start guide <../../get_started/quick_start_guide/installation>`.)

.. code-block::

  (otx) ...$ otx train otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                      --train-data-roots data/flower_photos \
                      --unlabeled-data-roots tests/assets/imagenet_dataset \
                      params --algo_backend.train_type SEMISUPERVISED

In the train log, you can check that the train type is set to **SEMISUPERVISED** and related configurations are properly loaded as following:

.. code-block::

  ...
  2023-02-22 06:21:54,492 | INFO : called _init_recipe()
  2023-02-22 06:21:54,492 | INFO : train type = SEMISUPERVISED
  2023-02-22 06:21:54,492 | INFO : train type = SEMISUPERVISED - loading training_extensions/otx/recipes/stages/classification/semisl.yaml
  2023-02-22 06:21:54,500 | INFO : Replacing runner from EpochRunnerWithCancel to EpochRunnerWithCancel.
  2023-02-22 06:21:54,503 | INFO : initialized recipe = training_extensions/otx/recipes/stages/classification/semisl.yaml
  ...


After training ends, a trained model is saved in ``otx-workspace-CLASSIFICATION/models`` since OTX generates a workspace named ``otx-workspace-CLASSIFICATION`` in default.

2. We can evaluate the trained model with auto-splitted validation dataset in the workspace and 
save results to ``otx-workspace-CLASSIFICATION/performance.json`` by the following command:

.. code-block::

  (otx) ...$ otx eval otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml \
                      --test-data-roots otx-workspace-CLASSIFICATION/splitted_dataset/val \
                      --load-weights otx-workspace-CLASSIFICATION/models/weights.pth \
                      --save-performance otx-workspace-CLASSIFICATION/performance.json
