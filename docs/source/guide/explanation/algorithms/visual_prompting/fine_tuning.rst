Visual Prompting (Fine-tuning)
==================================

Visual prompting is a computer vision task that uses a combination of an image and prompts, such as texts, bounding boxes, points, and so on to troubleshoot problems.
Using these useful prompts, the main purpose of this task is to obtain labels from unlabeled datasets, and to use generated label information on particular domains or to develop a new model with the generated information.

This section examines the solutions for visual prompting offered by the OpenVINO Training Extensions library.
`Segment Anything (SAM) <https://arxiv.org/abs/2304.02643>`_, is one of the most famous visual prompting methods and this model will be used to adapt a new dataset domain.
Because `SAM <https://arxiv.org/abs/2304.02643>`_ was trained by using web-scale dataset and has huge backbone network, fine-tuning the whole network is difficult and lots of resources are required.
Therefore, in this section, we try to fine-tune the mask decoder only for several epochs to increase performance on the new dataset domain.
For fine-tuning `SAM <https://arxiv.org/abs/2304.02643>`_, we use following algorithms components:

.. _visual_prompting_finetuning_pipeline:

- ``Pre-processing``: Resize an image according to the longest axis and pad the rest with zero.

- ``Optimizer``: We use `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer.

- ``Loss function``: We use standard loss combination, 20 * focal loss + dice loss + iou loss, used in `SAM <https://arxiv.org/abs/2304.02643>`_ as it is.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting. Early stopping will be automatically applied.


.. note::

    In OTX 2.0, fine-tuning using either bounding boxes or points is available.
    Given both bounding boxes and points prompts at the same time, bounding boxes have priority because using bounding boxes as prompts is more accurate.

.. note::

    Currently, Post-Training Quantization (PTQ) for `SAM <https://arxiv.org/abs/2304.02643>`_ is only supported, not Quantization Aware Training (QAT).


**************
Dataset Format
**************
.. _visual_prompting_dataset:

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

We support four dataset formats for visual prompting:

- `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/common_semantic_segmentation.html>`_ for semantic segmentation

- `COCO <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/coco.html>`_ for instance segmentation

- `Pascal VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_ for instance segmentation and semantic segmentation

- `Datumaro <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/datumaro.html>`_ for custom format dataset


******
Models
******
.. _visual_prompting_model:

We support the following model recipes in experimental phase:

+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+---------------------+-----------------+
|                                                                                        Recipe ID                                                           |     Name     | Complexity (GFLOPs) | Model size (MB) |
+============================================================================================================================================================+==============+=====================+=================+
| `Visual_Prompting_SAM_Tiny_ViT <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/visual_prompting/sam_tiny_vit.yaml>`_   | SAM_Tiny_ViT | 38.55               | 47              |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+---------------------+-----------------+
| `Visual_Prompting_SAM_ViT_B <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/visual_prompting/sam_vit_b.yaml>`_         | SAM_ViT_B    | 454.76              | 363             |
+------------------------------------------------------------------------------------------------------------------------------------------------------------+--------------+---------------------+-----------------+

To check feasibility of `SAM <https://arxiv.org/abs/2304.02643>`_, we did experiments using three public datasets with each other domains: `WGISD <https://github.com/thsant/wgisd>`_ and `Trashcan <https://conservancy.umn.edu/handle/11299/214865>`_, and checked `F1-score <https://en.wikipedia.org/wiki/F-score>`_.
We used sampled training data from `Trashcan <https://conservancy.umn.edu/handle/11299/214865>`_ and full training data (=110) from `WGISD <https://github.com/thsant/wgisd>`_.

+---------------------------------------------------------------+--------------------+
|                            Dataset                            |      #samples      |
+===============================================================+====================+
| `WGISD <https://github.com/thsant/wgisd>`_                    | 110                |
+---------------------------------------------------------------+--------------------+
| `Trashcan <https://conservancy.umn.edu/handle/11299/214865>`_ | 500                |
+---------------------------------------------------------------+--------------------+

The below table shows performance improvement after fine-tuning.

+--------------+--------------------------------------------+---------------------------------------------------------------+
|  Model name  | `WGISD <https://github.com/thsant/wgisd>`_ | `Trashcan <https://conservancy.umn.edu/handle/11299/214865>`_ |
+==============+============================================+===============================================================+
| SAM_Tiny_ViT | 90.32 → 92.10 (+1.78)                      | 82.38 → 85.35 (+2.97)                                         |
+--------------+--------------------------------------------+---------------------------------------------------------------+
| SAM_ViT_B    | 92.32 → 93.16 (+0.84)                      | 79.61 → 86.40 (+6.79)                                         |
+--------------+--------------------------------------------+---------------------------------------------------------------+

According to datasets, ``learning rate`` and ``batch size`` can be adjusted like below:

.. code-block:: shell

    (otx) ...$ otx train \
        --config <model_config_path> \
        --data_root <path_to_data_root> \
        --data.config.train_subset.batch_size <batch_size_to_be_updated> \
        --optimizer.lr <learning_rate_to_be_updated>
