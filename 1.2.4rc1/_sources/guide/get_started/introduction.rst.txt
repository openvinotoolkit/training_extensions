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

OpenVINO™ Training Extensions will provide the :doc:`following features <../explanation/index>` in coming releases:

- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Half-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient :doc:`hyper-parameter optimization module <../explanation/additional_features/hpo>` (**HPO**). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- OpenVINO™ Training Extensions uses `Datumaro <https://openvinotoolkit.github.io/datumaro/docs/>`_ as the backend to handle datasets. On account of that, OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. In the future there will be more supported formats available to give more freedom of datasets format choice.
- Improved :doc:`auto-configuration functionality <../explanation/additional_features/auto_configuration>`. OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model template to provide the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.
