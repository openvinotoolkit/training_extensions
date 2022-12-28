Hierarchical Classification
=========

*********
Dataset format
*********
The goal of this task is to predict the hierarchical tree of the labels. Assume we have a label tree as below:

.. image:: ../../../utils/images/label_tree.png
  :width: 600
  :alt: image uploaded from this `source <https://towardsdatascience.com/https-medium-com-noa-weiss-the-hitchhikers-guide-to-hierarchical-classification-f8428ea1e076>`_

The goal of our algorithm is to return the right branch of this tree. For example: ``Persian -> Cats -> Pets``


This type of algorithm is not so common in the computer vision field and doesn't have academic datasets for research. Since that, we created our dataset
format that is supported by `Datumaro <https://github.com/openvinotoolkit/datumaro>`_. An example of the annotations format and dataset structure can be found in our `sample <https://github.com/openvinotoolkit/training_extensions/tree/feature/otx/data/datumaro/datumaro_h-label>`_

Besides that we support the following dataset format:

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
| An example of annotations structure can be found `here <https://github.com/openvinotoolkit/training_extensions/blob/feature/otx/data/car_tree_bug/annotations/hierarchical_default.json>`_. For more intuition below the hierarchical structure represented for the image example given above:

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


*********
Backbones
*********

We use the same model templates as for multi-class Classification. Please, refer: :doc:`multi_class_classification`

Supervised Incremental Learning
*********

We solve this task by assigning a separate head for each label group on each hierarchical level. Concretely, we have two types of label groups:

- **Exclusive label group**, where the classifier solves multi-class classification problem and assigns only one label from the tree level.\

- **Inclusive label group**, where the classifier solves multi-label classification problem and can assign multiple labels from the tree level.


In this fashion, we train different classifiers, one - for each tree level. We use the same training strategy as for :doc:`multi_class_classification` task.


| In the inference stage, we traverse the tree from head to leaves and obtain labels predicted by the corresponding classifier.
| Let's say we forward an image with the label tree pictured in the above section. On the first level, our corresponding classifier returns 3 predictions.
| We perform an *argmax* operation and obtain, for example, class ``Cats``. Then, we choose a classifier related to ``{Siamse, Persian, Sphynx}`` label group, obtain its predictions, and after performing the *argmax* operation we choose our last leaf label.
| After that, we can easily reconstruct the final predicted tree branch: ``Persian -> Cats -> Pets``