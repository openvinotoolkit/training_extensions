Hierarchical Classification
===========================

**************
Dataset Format
**************

The goal of this task is to predict the hierarchical tree of the labels. Assume we have a label tree as below:

.. _hierarchical_image_example:

.. image:: ../../../../utils/images/label_tree.png
  :width: 600
  :alt: image uploaded from this `source <https://towardsdatascience.com/https-medium-com-noa-weiss-the-hitchhikers-guide-to-hierarchical-classification-f8428ea1e076>`_

The goal of our algorithm is to return the right branch of this tree. For example: ``Persian -> Cats -> Pets``

For hierarchical image classificaiton we created our custom dataset format that is supported by `Datumaro <https://github.com/openvinotoolkit/datumaro>`_. An example of the annotations format and dataset structure can be found in our `sample <https://github.com/openvinotoolkit/training_extensions/tree/feature/otx/data/datumaro/datumaro_h-label>`_.

Besides that we support the following custom dataset format:

::

    data
    ├── images
        ├── train
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
        ├── val
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
    └── annotations
        ├── train.json
        └── val.json

| Where annotations ``*.json`` consists of two keys: **"images"** and **"hierarchy"**. Key **"images"** includes lists of unique images with lists of classes presented on this image. Key **"hierarchy"** describes each label group presented in the label tree.
| An example of annotations structure can be found `here <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/data/car_tree_bug/annotations/hierarchical_default.json>`_. For more intuition below is the hierarchical structure represented for the given above :ref:`image example <hierarchical_image_example>`:

::

  "hierarchy": [
    {
      "parent": "self",
      "group": "Pets",
      "labels": ["Dogs", "Cats", "Unicorns"],
      "task_type": "single-label"
    },
    {
      "parent": "Pets",
      "group": "Cats",
      "labels": ["Siamse, Persian, Sphynx"],
      "task_type": "single-label"
    },
    {
      "parent": "Pets",
      "group": "Dogs",
      "labels": ["Poodle, French Bulldog, Dalmatian, Labrador"],
      "task_type": "single-label"
    },
    {
      "parent": "Pets",
      "group": "Unicorns",
      "labels": ["Pegasus, Rainbow Unicorn, Narwhal"],
      "task_type": "single-label"
    }
  ]

To use OTX with this format is required to pass annotations files paths and images dataset root paths directly to CLI command:

.. code-block::

    $ otx {train, optimize} <model_template> --train-ann-files /path/to/train_annotation_file.json --val-ann-files /path/to/val_annotation_file.json --train-data-roots /path/to/train/images_folder --val-data-roots /path/to/val/images_folder
    $ otx eval <model_template> --test-ann-files /path/to/test_annotation_file.json --test-data-roots /path/to/test/images_folder --load-weights path/to/weights

******
Models
******

We use the same model templates as for Multi-class Classification. Please, refer: :ref:`Classificaiton Models <classificaiton_models>`.

*******************
Supervised Learning
*******************

| We solve this task by assigning a separate head for each label group on each hierarchical level. Spicifically, we have a classifier that solves multi-class classification problem and assigns one label from the given exclusive label group.
| To create non-exclusive label group we can construct single-label exclusive groups for every label and each of them will be handled by an individual binary classifier.
| In this fashion, we train different classifiers, one - for each label group. We use the same training strategy as for :doc:`multi_class_classification` task.


| At the inference stage, we traverse the tree from head to leaves and obtain labels predicted by the corresponding classifier.
| Let's say we forward an image with the label tree pictured in the :ref:`dataset format <hierarchical_image_example>` section. On the first level, our corresponding classifier returns 3 predictions.
| We perform an *argmax* operation and obtain, for example, class ``Cats``. Then, we choose a classifier related to ``{Siamse, Persian, Sphynx}`` label group, obtain its predictions, and after performing the *argmax* operation we choose our last leaf label.
| After that, we can easily reconstruct the final predicted tree branch: ``Persian -> Cats -> Pets``.

********************
Incremental Learning
********************

To be added soon