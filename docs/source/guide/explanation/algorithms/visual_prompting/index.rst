Visual Prompting
=================

Visual prompting is a computer vision task that uses a combination of an image and prompts, such as texts, bounding boxes, points, and so on to troubleshoot problems.
Using these useful prompts, the main purpose of this task is to obtain labels from unlabeled datasets, and to use generated label information on particular domains or to develop a new model with the generated information.

This section examines the solutions for visual prompting offered by the OpenVINO Training Extensions library.
`Segment Anything (SAM) <https://arxiv.org/abs/2304.02643>`_, is one of the most famous visual prompting methods and this model will be used to adapt a new dataset domain.
Because `SAM <https://arxiv.org/abs/2304.02643>`_ was trained by using web-scale dataset and has huge backbone network, fine-tuning the whole network is difficult and lots of resources are required.
Therefore, in this section, we try to fine-tune only mask decoder only for several epochs to increase performance on the new dataset domain.
For fine-tuning `SAM <https://arxiv.org/abs/2304.02643>`_, we use following algorithms components:

.. _visual_prompting_finetuning_pipeline:

- ``Pre-processing``: Resize an image according to the longest axis and pad the rest with zero.

- ``Optimizer``: We use `Adam <https://arxiv.org/abs/1412.6980>`_ optimizer with 1e-6 learning rate.

- ``Loss function``: We use standard loss combination, 20 * focal loss + dice loss + iou loss, used in `SAM <https://arxiv.org/abs/2304.02643>`_ as it is.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting. Early stopping will be automatically applied.


**************
Dataset Format
**************
.. _visual_prompting_dataset:

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_.

We support three dataset formats for visual prompting:

- `Common Semantic Segmentation <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/common_semantic_segmentation.html>`_ for semantic segmentation

- `COCO <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/coco.html>`_ for instance segmentation

- `Pascal VOC <https://openvinotoolkit.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_ for instance segmentation and semantic segmentation


If you organized supported dataset format, starting training will be very simple. We just need to pass a path to the root folder and desired model template to start training:

.. code-block::

    $ otx train <model_template> \
        --train-data-roots <path_to_data_root> \
        --val-data-roots <path_to_data_root>

.. note::

    During training, mDice for binary mask without label information is used for train/validation metric.
    After training, if using ``otx eval`` to evaluate performance, mDice for binary or multi-class masks with label information will be used.
    As you can expect, performance will be different between ``otx train`` and ``otx eval``, but if unlabeled mask performance is high, labeld mask performance is high as well.


******
Models
******
.. _visual_prompting_model:

We support the following model templates in experimental phase:
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+---------------------+-----------------+
| Template ID                                                                                                                                                                          | Name      | Complexity (GFLOPs) | Model size (MB) |
+======================================================================================================================================================================================+===========+=====================+=================+
| `Visual_Prompting_SAM_ViT_B <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/visual_prompting/configs/sam_vit_b/template_experimental.yaml>`_ | SAM_ViT_B | 487                 | 374             |
+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-----------+---------------------+-----------------+

In the table below the `Dice score <https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient>`_ on some academic datasets.
The below table shows performance improvement after fine-tuning.

+---------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+
| Model name    | `DIS5K <https://xuebinqin.github.io/dis/index.html>`_        | `Cityscapes <https://www.cityscapes-dataset.com/>`_ | `Pascal-VOC 2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ | `KITTI full <https://www.cvlibs.net/datasets/kitti/index.php>`_ | Mean   |
+===============+==============================================================+=====================================================+======================================================================+=================================================================+========+
| SAM_ViT_B     | 79.95                                                        | 62.38                                               | 58.26                                                                | 36.06                                                           | 59.16  |
+-----------------------+--------------------------------------------------------------+-----------------------------------------------------+----------------------------------------------------------------------+-----------------------------------------------------------------+--------+



