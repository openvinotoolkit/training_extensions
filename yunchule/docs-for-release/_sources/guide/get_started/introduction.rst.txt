.. raw:: html

   <div style="margin-bottom:30px;">
   <img src="../../_static/logos/otx-logo-black.png" alt="Logo" width="900" style="display:block;margin:auto;">
   </div>

Introduction
============

OpenVINO™ Training Extensions is a command-line interface (CLI) framework designed for low-code computer vision deep-learning model training. OpenVINO™ Training Extensions lets users train, infer, optimize and deploy models simply and fast even with low expertise in the deep learning field. OpenVINO™ Training Extensions offers a diverse combination of model architectures, learning methods, and training types using PyTorch and OpenVINO™ toolkit. OpenVINO™ Training Extensions provides so-called "model templates" for every supported task which have been tested on various datasets and are a turnkey solution for obtaining an average good model without the need to change any hyperparameters. Besides, it is possible to configure your own model based on torchvision, mmcv, pytorchcv and OpenVINO Model Zoo (OMZ). Moreover, OpenVINO™ Training Extensions supports auto-configuration functionality to choose a suitable model template based on the dataset. We will further extend our functionality to make training as much simple as possible for obtaining accurate, fast and light models ready to integrate into your projects.

To this end OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box support
- **Semantic segmentation**
- **Instance segmentation** including tiling algorithm support
- **Action recognition** including action classification and detection
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

OpenVINO™ Training Extensions also supports different :doc:`training types <../explanation/algorithms/index>`:

- **Supervised**, incremental training including class incremental scenario and contrastive learning for classification and semantic segmentation tasks
- **Semi-supervised learning**
- **Self-supervised learning**

Moving forward, OpenVINO™ Training Extensions provides the :doc:`following features <../explanation/index>`:

- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Half-precision training** to save GPUs memory and use larger batch sizes
- Integrated, efficient :doc:`hyper-parameter optimization module <../explanation/additional_features/hpo>` (HPO). Through dataset proxy and built-in hyper-parameter optimizer, you can get much faster hyper-parameter optimization compared to other off-the-shelf tools. The hyperparameter optimization is dynamically scheduled based on your resource budget.
- OpenVINO™ Training Extensions uses **Datumaro** as the backend to hadle datasets. Thanks to that, OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. We constantly working to extend supported formats to give more freedom of datasets format choice.
- :doc:`Auto-configuration functionality <../explanation/additional_features/auto_configuration>`. OpenVINO™ Training Extensions analyzes provided dataset and chooses the proper task and model template to have the best accuracy/speed trade-off. It will also make a random auto-split of your dataset if there is no validation set provided.
