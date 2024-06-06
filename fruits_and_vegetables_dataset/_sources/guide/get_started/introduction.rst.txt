.. raw:: html

    <div style="margin-bottom:30px;">
    <img src="../../_static/logos/otx-logo-black.png" alt="Logo" width="900" style="display:block;margin:auto;background-color:white;">
    </div>

Introduction
============

**OpenVINO™ Training Extensions** is a low-code transfer learning framework for Computer Vision.

The CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field. OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on `PyTorch <https://pytorch.org/>`_ , `Lightning <https://lightning.ai/>`_ and `OpenVINO™ toolkit <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html>`_.

OpenVINO™ Training Extensions provide `recipe <https://github.com/openvinotoolkit/training_extensions/tree/develop/src/otx/recipe>`_ for every supported task type, which consolidates necessary information to build a model. Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general. If you are an experienced user, you can configure your own model based on `torchvision <https://pytorch.org/vision/stable/index.html>`_, `mmcv <https://github.com/open-mmlab/mmcv>`_ and `OpenVINO Model Zoo (OMZ) <https://github.com/openvinotoolkit/open_model_zoo>`_ frameworks.

Furthermore, OpenVINO™ Training Extensions provides :doc:`automatic configuration <../explanation/additional_features/auto_configuration>` of task types and hyperparameters. The framework will identify the most suitable recipe based on your dataset, and choose the best hyperparameter configuration. The development team is continuously extending functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

************
Key Features
************

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box support
- **Semantic segmentation**
- **Instance segmentation** including tiling algorithm support
- **Action recognition** including action classification and detection
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation
- **Visual Prompting** tasks including segment anything model, zero-shot visual prompting

OpenVINO™ Training Extensions supports the :doc:`following learning methods <../explanation/algorithms/index>`:

- **Supervised**, incremental training, which includes class incremental scenario.

OpenVINO™ Training Extensions will provide the :doc:`following features <../explanation/additional_features/index>` in coming releases:

- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Half-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient :doc:`hyper-parameter optimization module <../explanation/additional_features/hpo>` (**HPO**). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- OpenVINO™ Training Extensions uses `Datumaro <https://openvinotoolkit.github.io/datumaro/stable/index.html>`_ as the backend to handle datasets. On account of that, OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. In the future there will be more supported formats available to give more freedom of datasets format choice.
- Improved :doc:`auto-configuration functionality <../explanation/additional_features/auto_configuration>`. OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model recipe to provide the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.

*********************
Documentation content
*********************

1. :octicon:`light-bulb` **Quick start guide**:

.. grid::
    :gutter: 1

    .. grid-item-card:: :octicon:`package` Installation Guide
        :link: installation
        :link-type: doc
        :text-align: center

        Learn more about how to install OpenVINO™ Training Extensions

    .. grid-item-card:: :octicon:`code-square` API Quick-Guide
        :link: api_tutorial
        :link-type: doc
        :text-align: center

        Learn more about how to use OpenVINO™ Training Extensions Python API.

    .. grid-item-card:: :octicon:`terminal` CLI Guide
        :link: cli_commands
        :link-type: doc
        :text-align: center

        Learn more about how to use OpenVINO™ Training Extensions CLI commands

2. :octicon:`book` **Tutorials**:

.. grid:: 1 2 2 3
    :margin: 1 1 0 0
    :gutter: 1

    .. grid-item-card:: Classification
        :link: ../tutorials/base/how_to_train/classification
        :link-type: doc
        :text-align: center

        Learn how to train a classification model

    .. grid-item-card:: Detection
        :link: ../tutorials/base/how_to_train/detection
        :link-type: doc
        :text-align: center

        Learn how to train a detection model.

    .. grid-item-card:: Instance Segmentation
        :link: ../tutorials/base/how_to_train/instance_segmentation
        :link-type: doc
        :text-align: center

        Learn how to train an instance segmentation model

    .. grid-item-card:: Semantic Segmentation
        :link: ../tutorials/base/how_to_train/semantic_segmentation
        :link-type: doc
        :text-align: center

        Learn how to train a semantic segmentation model

    .. grid-item-card:: Anomaly Task
        :link: ../tutorials/base/how_to_train/anomaly_detection
        :link-type: doc
        :text-align: center

        Learn how to train an anomaly detection model

    .. grid-item-card:: Action Classification
        :link: ../tutorials/base/how_to_train/action_classification
        :link-type: doc
        :text-align: center

        Learn how to train an action classification model

    .. grid-item-card:: Action Detection
        :link: ../tutorials/base/how_to_train/action_detection
        :link-type: doc
        :text-align: center

        Learn how to train an action detection model

    .. grid-item-card:: Visual Prompting
        :link: ../tutorials/base/how_to_train/visual_prompting
        :link-type: doc
        :text-align: center

        Learn how to train a visual prompting model

    .. grid-item-card:: Advanced
        :link: ../tutorials/advanced/index
        :link-type: doc
        :text-align: center

        Learn how to use advanced features of OpenVINO™ Training Extensions

3. **Explanation section**:

This section consists of an algorithms explanation and describes additional features that are supported by OpenVINO™ Training Extensions.
:ref:`Algorithms <algo_section_ref>` section includes a description of all supported algorithms:

   1. Explanation of the task and main supervised training pipeline.
   2. Description of the supported datasets formats for each task.
   3. Available recipes and models.
   4. Incremental learning approach.

:ref:`Additional Features <features_section_ref>` section consists of:

   1. Overview of model optimization algorithms.
   2. Hyperparameters optimization functionality (HPO).
   3. Auto-configuration algorithm to select the most appropriate training pipeline for a given dataset.

4. **Reference**:

This section gives an overview of the OpenVINO™ Training Extensions code base. There source code for Entities, classes and functions can be found.

5. **Release Notes**:

There can be found a description of new and previous releases.
