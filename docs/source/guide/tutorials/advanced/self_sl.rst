############################
Use Self-Supervised Learning
############################

This tutorial introduces how to train a model using self-supervised learning and how to fine-tune the model with pre-trained weights.
OpenVINO™ Training Extensions provides self-supervised learning methods for :doc:`multi-classification <../../explanation/algorithms/classification/multi_class_classification>` and :doc:`semantic segmentation <../../explanation/algorithms/segmentation/semantic_segmentation>`.

The process has been tested on the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.7

.. note::

    This example demonstrates how to work with :ref:`self-supervised learning for classification <selfsl_multi_class_classification>`.
    There are some differences between classfication and semantic segmentation, so there will be some notes for :ref:`self-supervised learning for semantic segmentation <selfsl_semantic_segmentation>`.

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../get_started/quick_start_guide/installation>` 
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual 
environment:

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

************
Pre-training
************

1. Prepare dataset and model. To prepare dataset and decide model, refer to :doc:`classification tutorial <../base/how_to_train/classification>`.
In this self-supervised learning tutorial, `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_ and :ref:`MobileNet-V3-large-1x <classification_models>` model used in :doc:`classification tutorial <../base/how_to_train/classification>` is used as it is.

2. Prepare OpenVINO™ Training Extensions workspace for **supervised learning** by running 
the following command:

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
    [*] Update data configuration file to: otx-workspace-CLASSIFICATION/data.yaml

3. Prepare an OpenVINO™ Training Extensions workspace 
for **self-supervised learning** by running the following command:

.. code-block::

    (otx) ...$ otx build --train-data-roots data/flower_photos --model MobileNet-V3-large-1x --train-type SELFSUPERVISED --workspace otx-workspace-CLASSIFICATION-Selfsupervised

    [*] Workspace Path: otx-workspace-CLASSIFICATION-Selfsupervised
    [*] Load Model Template ID: Custom_Image_Classification_MobileNet-V3-large-1x
    [*] Load Model Name: MobileNet-V3-large-1x[*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/selfsl/model.py
    [*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/selfsl/data_pipeline.py
    [*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/deployment.py
    [*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/hpo_config.yaml
    [*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/model_hierarchical.py
    [*]     - Updated: otx-workspace-CLASSIFICATION-Selfsupervised/model_multilabel.py
    [*] Update data configuration file to: otx-workspace-CLASSIFICATION-Selfsupervised/data.yaml

.. note::

    Three things must be considered to set the workspace for self-supervised learning:

    1. add ``--train-type Selfsupervised`` in the command to get the training components for self-supervised learning,
    2. update the path set as ``train-data-roots``,
    3. and add ``--workspace`` to distinguish self-supervised learning workspace from supervised learning workspace.

After the workspace creation, the workspace structure is as follows:

.. code-block::

    otx-workspace-CLASSIFICATION
    ├── compression_config.json
    ├── configuration.yaml
    ├── data_pipeline.py
    ├── data.yaml
    ├── deployment.py
    ├── hpo_config.yaml
    ├── model_hierarchical.py
    ├── model_multilabel.py
    ├── model.py
    ├── splitted_dataset
    │   ├── train
    │   └── val
    └── template.yaml
    otx-workspace-CLASSIFICATION-Selfsupervised
    ├── configuration.yaml
    ├── data.yaml
    ├── deployment.py
    ├── hpo_config.yaml
    ├── model_hierarchical.py
    ├── model_multilabel.py
    ├── selfsl
    │   ├── data_pipeline.py
    │   └── model.py
    └── template.yaml

.. note::

    For :ref:`semantic segmentation <selfsl_semantic_segmentation>`, ``--train-data-root`` must be set to a directory including only images, not masks, like below.
    
    For `VOC2012 dataset <http://host.robots.ox.ac.uk/pascal/VOC/voc2012>`_ used in :doc:`semantic segmentation tutorial <../base/how_to_train/semantic_segmentation>`, for example, the path ``data/VOCdevkit/VOC2012/JPEGImages`` must be set instead of ``data/VOCdevkit/VOC2012``.
    
    Please refer to :ref:`Explanation of Self-Supervised Learning for Semantic Segmentation <selfsl_semantic_segmentation>`.
    And don't forget to add ``--train-type Selfsupervised``.

    .. code-block::

        (otx) ...$ otx build --train-data-roots data/VOCdevkit/VOC2012/JPEGImages \
                            --model Lite-HRNet-18-mod2 \
                            --train-type Selfsupervised

4. To start training we need to call ``otx train`` 
command in **self-supervised learning** workspace:

.. code-block::

    (otx) ...$ cd otx-workspace-CLASSIFICATION-Selfsupervised
    (otx) ...$ otx train --data ../otx-workspace-CLASSIFICATION/data.yaml
    
    ...

    2023-02-23 19:41:36,879 | INFO : Iter [4970/5000]       lr: 8.768e-05, eta: 0:00:29, time: 1.128, data_time: 0.963, memory: 7522, current_iters: 4969, loss: 0.2788
    2023-02-23 19:41:46,371 | INFO : Iter [4980/5000]       lr: 6.458e-05, eta: 0:00:19, time: 0.949, data_time: 0.782, memory: 7522, current_iters: 4979, loss: 0.2666
    2023-02-23 19:41:55,806 | INFO : Iter [4990/5000]       lr: 5.037e-05, eta: 0:00:09, time: 0.943, data_time: 0.777, memory: 7522, current_iters: 4989, loss: 0.2793
    2023-02-23 19:42:05,105 | INFO : Saving checkpoint at 5000 iterations
    2023-02-23 19:42:05,107 | INFO : ----------------- BYOL.state_dict_hook() called
    2023-02-23 19:42:05,314 | WARNING : training progress 100%
    2023-02-23 19:42:05,315 | INFO : Iter [5000/5000]       lr: 4.504e-05, eta: 0:00:00, time: 0.951, data_time: 0.764, memory: 7522, current_iters: 4999, loss: 0.2787
    2023-02-23 19:42:05,319 | INFO : run task done.
    2023-02-23 19:42:05,323 | INFO : called save_model
    2023-02-23 19:42:05,498 | INFO : Final model performance: Performance(score: -1, dashboard: (6 metric groups))
    2023-02-23 19:42:05,499 | INFO : train done.
    [*] Save Model to: models

.. note::
    To use the same splitted train dataset, set ``--data ../otx-workspace-CLASSIFICATION/data.yaml`` insead of using ``data.yaml`` in self-supervised learning workspace.

The training will return artifacts: ``weights.pth`` and ``label_schema.json`` and we can use the weights to fine-tune the model using the target dataset.
The final model performance will be set to -1, but it doesn't matter because self-supervised learning doesn't use accuracy.
Let's see how to fine-tune the model using pre-trained weights below.

***********
Fine-tuning
***********

After pre-training progress, start fine-tuning by calling the below command with adding ``--load-weights`` argument in supervised learning workspace.

.. code-block::

    (otx) ...$ cd ../otx-workspace-CLASSIFICATION
    (otx) ...$ otx train --load-weights ../otx-workspace-CLASSIFICATION-Selfsupervised/models/weights.pth

    ...

    2023-02-23 20:56:24,307 | INFO : run task done.
    2023-02-23 20:56:28,883 | INFO : called evaluate()
    2023-02-23 20:56:28,895 | INFO : Accuracy after evaluation: 0.9604904632152589
    2023-02-23 20:56:28,896 | INFO : Evaluation completed
    Performance(score: 0.9604904632152589, dashboard: (3 metric groups))

For comparison, we can also obtain the performance without pre-trained weights as below:

.. code-block::

    (otx) ...$ otx train

    ...

    2023-02-23 18:24:34,453 | INFO : run task done.
    2023-02-23 18:24:39,043 | INFO : called evaluate()
    2023-02-23 18:24:39,056 | INFO : Accuracy after evaluation: 0.9550408719346049
    2023-02-23 18:24:39,056 | INFO : Evaluation completed
    Performance(score: 0.9550408719346049, dashboard: (3 metric groups))

With self-supervised learning, we can obtain well-adaptive weights and train the model more accurately.
This example showed a little improvement (0.955 → 0.960), but if we use only a few samples that are *too difficult to train a model on*, then
self-supervised learning can be the solution to improve the model.
You can check performance improvement examples in :ref:`self-supervised learning for classification <selfsl_multi_class_classification>` documentation.

.. note::
    Then we obtain the new model after fine-tuning, we can proceed with optimization and exporting as described in :doc:`classification tutorial <../base/how_to_train/classification>`.
