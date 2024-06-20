############################
Use Semi-Supervised Learning
############################

This tutorial provides an example of how to use semi-supervised learning with OpenVINO™ Training Extensions on the specific dataset.

OpenVINO™ Training Extensions now offers semi-supervised learning, which combines labeled and unlabeled data during training to improve model accuracy in case when we have a small amount of annotated data. Currently, this type of training is available for multi-class classification.

If you want to learn more about the algorithms used in semi-supervised learning, please refer to the explanation section below:

- `Multi-class Classification <../../explanation/algorithms/classification/multi_class_classification.html#semi-supervised-learning>`__

In this tutorial, we use the MobileNet-V3-large model for multi-class classification to cite an example of semi-supervised learning.

The process has been tested on the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8

.. note::

    To learn how to export the trained model, refer to `classification export <../base/how_to_train/classification.html#export>`__.

    To learn how to optimize the trained model (.xml) with OpenVINO™ PTQ, refer to `classification optimization <../base/how_to_train/classification.html#optimization>`__.

This tutorial explains how to train a model in semi-supervised learning mode and how to evaluate the resulting model.

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block:: shell

    .otx/bin/activate
    # or by this line, if you created an environment, using tox
    . venv/otx/bin/activate

***************************
Dataset preparation
***************************

We use the same dataset, `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_, as we do in :doc:`classification tutorial <../base/how_to_train/classification>`.

Since it is assumed that we have additional unlabeled images,
we make a use of ``tests/assets/classification_semisl_dataset/unlabeled`` for this purpose as an example.

please keep the exact same name for the train/val/test folder, to identify the dataset.

.. code-block:: shell

    flower_photos
    ├──labeled
    |    ├──train
    |    |    ├── daisy
    |    |    ├── dandelion
    |    |    ├── roses
    |    |    ├── sunflowers
    |    |    ├── tulips
    |    ├──val
    |    |    ├── daisy
    |    |    ├── ...
    |    ├──test
    |    |    ├── daisy
    |    |    ├── ...
    ├──unlabeled


*********
Training
*********

1. The recipe that provides Semi-SL can be found below.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

                (otx) ...$ otx find --task MULTI_CLASS_CLS --pattern semisl
                ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃ Task            ┃ Model Name                            ┃ Recipe Path                                                                    ┃
                ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
                │ MULTI_CLASS_CLS │ tv_efficientnet_v2_l_semisl           │ src/otx/recipe/classification/multi_class_cls/tv_efficientnet_v2_l_semisl.yaml │
                │ MULTI_CLASS_CLS │ mobilenet_v3_large_semisl             │ src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large_semisl.yaml   │
                │ MULTI_CLASS_CLS │ efficientnet_b0_semisl                │ src/otx/recipe/classification/multi_class_cls/efficientnet_b0_semisl.yaml      │
                │ MULTI_CLASS_CLS │ tv_efficientnet_b3_semisl             │ src/otx/recipe/classification/multi_class_cls/tv_efficientnet_b3_semisl.yaml   │
                │ MULTI_CLASS_CLS │ efficientnet_v2_semisl                │ src/otx/recipe/classification/multi_class_cls/efficientnet_v2_semisl.yaml      │
                │ MULTI_CLASS_CLS │ deit_tiny_semisl                      │ src/otx/recipe/classification/multi_class_cls/deit_tiny_semisl.yaml            │
                │ MULTI_CLASS_CLS │ dino_v2_semisl                        │ src/otx/recipe/classification/multi_class_cls/dino_v2_semisl.yaml              │
                │ MULTI_CLASS_CLS │ tv_mobilenet_v3_small_semisl          │ src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small_semisl.yaml│
                └─────────────────┴─────────────────────━━━━━━━━━━━━━─────┴────────────────────────────────────────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="MULTI_CLASS_CLS", pattern="*semisl")
            print(model_lists)
            '''
            [
                'tv_efficientnet_b3_semisl',
                'efficientnet_b0_semisl',
                'efficientnet_v2_semisl',
                ...
            ]
            '''

2. We will use the MobileNet-V3-large model for multi-class classification in semi-supervised learning mode.

.. tab-set::

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train \
                        --config src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large_semisl.yaml \
                        --data_root data/flower_photos/labeled \
                        --data.config.unlabeled_subset.data_root data/flower_photos/unlabeled

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.engine import Engine

            data_root = "data/flower_photos"
            recipe = "src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large_semisl.yaml"
            overrides = {"data.config.unlabeled_subset.data_root": "data/flower_photos/unlabeled"}

            engine = Engine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                      **kwargs,
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.core.config.data import DataModuleConfig, UnlabeledDataConfig
            from otx.core.data.module import OTXDataModule
            from otx.engine import Engine

            data_config = DataModuleConfig(..., unlabeled_subset=UnlabeledDataConfig(data_root="data/flower_photos/unlabeled", ...))
            datamodule = OTXDataModule(..., config=data_config)

            engine = Engine(..., datamodule=datamodule)

            engine.train(max_epochs=200)

The rest of the commands are the same as the original Classification tutorial.
Please refer to the :doc:`classification tutorial <../base/how_to_train/classification>` for more details.
