Visual Prompting (Zero-shot learning)
=====================================

Visual prompting is a computer vision task that uses a combination of an image and prompts, such as texts, bounding boxes, points, and so on to troubleshoot problems.
Using these useful prompts, the main purpose of this task is to obtain labels from unlabeled datasets, and to use generated label information on particular domains or to develop a new model with the generated information.

This section examines the solutions for visual prompting offered by the OpenVINO Training Extensions library.
`Segment Anything (SAM) <https://arxiv.org/abs/2304.02643>`_, is one of the most famous visual prompting methods and this model will be used to adapt a new dataset domain.
Especially, in this section, we try to automatically predict given images without any training, called as ``zero-shot learning``.
Unlike fine-tuning, zero-shot learning needs only pre-processing component.


.. _visual_prompting_zeroshot_pipeline:

- ``Pre-processing``: Resize an image according to the longest axis and pad the rest with zero.


.. note::

    Currently, zero-shot learning with `SAM <https://arxiv.org/abs/2304.02643>`_ with bounding boxes in the OpenVINO Training Extensions is only supported.
    We will support zero-shot learning with other prompts (points and texts) in the near future.

.. note::

    Currently, Post-Training Quantization (PTQ) for `SAM <https://arxiv.org/abs/2304.02643>`_ is only supported, not Quantization Aware Training (QAT).


**************
Dataset Format
**************
.. _visual_prompting_dataset:

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

We support three dataset formats for visual prompting:

- `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/common_semantic_segmentation.html>`_ for semantic segmentation

- `COCO <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/coco.html>`_ for instance segmentation

- `Pascal VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_ for instance segmentation and semantic segmentation


******
Models
******
.. _visual_prompting_zero_shot_model:

We support the following model templates in experimental phase:

+---------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+
|                                                                                          Template ID                                                          |          Name          | Complexity (GFLOPs) | Model size (MB) |
+===============================================================================================================================================================+========================+=====================+=================+
| `Zero_Shot_SAM_Tiny_ViT <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/zeto_shot_visual_prompting/sam_tiny_vit.yaml>`_   | Zero_Shot_SAM_Tiny_ViT | 38.18               | 25              |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------+---------------------+-----------------+

***************
Simple tutorial
***************
.. _visual_prompting_zero_shot_tutorial:

There are two steps for zero-shot inference: ``learn`` and ``infer``.
``Learn`` is to extracet reference features from given reference images and prompts. These extracted reference features will be used to get point candidates on given target images.
Extracted reference features will be saved in the model checkpoint (such as `weight.pth`) with the model.
You can do ``learn`` with the following source code:

.. code-block:: shell

    (otx) ...$ otx train --config <model_config_path> \
        --data_root <path_to_data_root>

``Infer`` is to get predicted masks on given target images. Unlike ``learn``, this stage doesn't need any prompt information.

.. code-block::

    (otx) ...$ otx test --config <model_config_path> \
        --data_root <path_to_data_root> \
        --checkpoint <path_to_weights_from_learn>


For example, when the positive (green) and the negative (red) points were given with the reference image for ``learn`` stage, you can get basic `SAM <https://arxiv.org/abs/2304.02643>`_ prediction result (left).
If you give the same reference image as the target image for ``infer`` stage, you can get target prediction results (right).

.. list-table::

    * - .. figure:: ../../../../../utils/images/vpm_ref_result.png

      - .. figure:: ../../../../../utils/images/vpm_ref_prediction.png


You can get target prediction results for other given images like below.

.. image:: ../../../../../utils/images/vpm_tgt_prediction.png
