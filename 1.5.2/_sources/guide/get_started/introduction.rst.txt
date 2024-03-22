.. raw:: html

   <div style="margin-bottom:30px;">
   <img src="../../_static/logos/otx-logo-black.png" alt="Logo" width="900" style="display:block;margin:auto;">
   </div>

Introduction
============

**OpenVINO™ Training Extensions** is a low-code transfer learning framework for Computer Vision.

The CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field. OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on `PyTorch <https://pytorch.org/>`_ and `OpenVINO™ toolkit <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html>`_.

OpenVINO™ Training Extensions provides a **“model template”** for every supported task type, which consolidates necessary information to build a model. Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general. If you are an experienced user, you can configure your own model based on `torchvision <https://pytorch.org/vision/stable/index.html>`_, `pytorchcv <https://github.com/osmr/imgclsmob>`_, `mmcv <https://github.com/open-mmlab/mmcv>`_ and `OpenVINO Model Zoo (OMZ) <https://github.com/openvinotoolkit/open_model_zoo>`_ frameworks.

Furthermore, OpenVINO™ Training Extensions provides :doc:`automatic configuration <../explanation/additional_features/auto_configuration>` of task types and hyperparameters. The framework will identify the most suitable model template based on your dataset, and choose the best hyperparameter configuration. The development team is continuously extending functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

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

OpenVINO™ Training Extensions supports the :doc:`following learning methods <../explanation/algorithms/index>`:

- **Supervised**, incremental training, which includes class incremental scenario and contrastive learning for classification and semantic segmentation tasks
- **Semi-supervised learning**
- **Self-supervised learning**

OpenVINO™ Training Extensions will provide the :doc:`following features <../explanation/additional_features/index>` in coming releases:

- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Half-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient :doc:`hyper-parameter optimization module <../explanation/additional_features/hpo>` (**HPO**). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- OpenVINO™ Training Extensions uses `Datumaro <https://openvinotoolkit.github.io/datumaro/stable/index.html>`_ as the backend to handle datasets. On account of that, OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. In the future there will be more supported formats available to give more freedom of datasets format choice.
- Improved :doc:`auto-configuration functionality <../explanation/additional_features/auto_configuration>`. OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model template to provide the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.

************
Documentation content
************

1. **Quick start guide**:

   1. Installation
   2. All possible OpenVINO™ Training Extensions CLI commands

2. **Tutorials**:

   This section reveals tutorials on how to use CLI for every supported task and training type.
   It provides the end-to-end solution from installation to model deployment and demo visualization on specific examples for each of the supported tasks.
   In the advanced section tutorial on how to use APIs instead of CLI is presented.

3. **Explanation section**:

   This section consists of an algorithms explanation and describes additional features that are supported by OpenVINO™ Training Extensions.
   :ref:`Algorithms <algo_section_ref>` section includes a description of all supported algorithms:

   1. Explanation of the task and main supervised training pipeline.
   2. Description of the supported datasets formats for each task.
   3. Available templates and models.
   4. Incremental learning approach.
   5. Semi-supervised and Self-supervised algorithms.

   :ref:`Additional Features <features_section_ref>` section consists of:

   1. Overview of model optimization algorithms.
   2. Hyperparameters optimization functionality (HPO).
   3. Auto-configuration algorithm to select the most appropriate training pipeline for a given dataset.

4. **Reference**:

   This section gives an overview of the OpenVINO™ Training Extensions code base. There source code for Entities, classes and functions can be found.

5. **Release Notes**:

   There can be found a description of new and previous releases.
