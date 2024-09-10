Training with Hugging Face pretrained vision models
===================================================

Overview
--------

OpenVINO™ Training Extensions streamlines your workflow by leveraging cutting-edge artificial intelligence technologies, enabling rapid and efficient model development. With this latest update, users can now easily train models for multi-class classification, object detection, and semantic segmentation tasks using pre-trained models from Hugging Face, and convert them into optimized Intermediate Representation (IR) models for OpenVINO.

Introduction to Hugging Face
-----------------------------

Hugging Face is an AI community and platform for machine learning that provides a vast repository of pre-trained models. These models span across various domains of natural language processing (NLP), computer vision, and more. Hugging Face's transformers library is widely recognized for its ease of use and state-of-the-art models, which can be fine-tuned for a wide range of tasks.

OpenVINO IR Model Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Model Optimization and Conversion**: Convert the trained models into the IR format using the OpenVINO toolkit to optimize performance across various hardware platforms.
- **Deployment Ready**: The converted models are ready for immediate deployment, and OpenVINO™ Training Extensions simplifies this process, supporting users in easily deploying models to edge devices or servers.

Getting Started
---------------

1. **Environment Setup**: Set up the OpenVINO™ Training Extensions environment and install the necessary libraries and toolkits.

You can follow the installation process from a :doc:`quick start guide <../../get_started/installation>` to create a universal virtual environment for OpenVINO™ Training Extensions.
Then install transformers with the command below.

.. code-block:: shell

    (otx) ...$ pip install transformers

2. **Dataset Preparation**: Set up dataset for training.

You can follow the preparation step per each tasks. :doc:`quick start guide <../base/how_to_train/index>`

3. **Model Selection and Training**: Select an appropriate pre-trained model from the Hugging Face hub and start training with your dataset using OpenVINO™ Training Extensions.

You can find a list of pre-trained models provided by hugging-face `here <https://huggingface.co/models>`_.
Currently, we support models in the Image Classification, Object Detection, and Image Segmentation categories, and some models may not be supported due to API policy.

OTX provides the model classes below in three tasks for this purpose:
- Classification: `otx.algo.classification.huggingface_model.HuggingFaceModelForMulticlassCls`
- Detection: `otx.algo.detection.huggingface_model.HuggingFaceModelForDetection`
- Semantic Segmentation: `otx.algo.segmentation.huggingface_model.HuggingFaceModelForSegmentation`

You can run training on a Custom Dataset using the examples below.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train \
                        --model otx.algo.classification.huggingface_model.HuggingFaceModelForMulticlassCls \
                        --model.model_name_or_path google/vit-base-patch16-224 \
                        --data_root /datasets/otx_v2_dataset/multiclass_classification/multiclass_food101_large \
                        --work_dir otx-workspace/vit-base-224

    .. tab-item:: API

        .. code-block:: python

            from otx.algo.classification.huggingface_model import HuggingFaceModelForMulticlassCls
            from otx.engine import Engine

            data_root = "/datasets/otx_v2_dataset/multiclass_classification/multiclass_food101_large"

            otx_model = HuggingFaceModelForMulticlassCls(
                model_name_or_path="google/vit-base-patch16-224",
                label_info=20,
            )

            engine = Engine(
                data_root=data_root,
                model=otx_model,
                work_dir="otx-workspace/vit-base-224",
            )

            engine.train()

.. note::

    For detection, you can use the `otx.algo.detection.huggingface_model.HuggingFaceModelForDetection` class,
    and for semantic segmentation, you can use the `otx.algo.segmentation.huggingface_model.HuggingFaceModelForSegmentation` class. Otherwise, the usage is the same.

4. **Model Conversion**: Convert the trained model into the IR format using the OpenVINO toolkit.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx export \
                        --work_dir otx-workspace/vit-base-224

    .. tab-item:: API

        .. code-block:: python

            engine.export()
