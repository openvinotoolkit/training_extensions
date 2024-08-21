Algorithms
==========

.. _algo_section_ref:

OpenVINO™ Training Extensions supports different training types to solve a variety of computer vision problems. This section provides what exactly we utilize inside our algorithms providing an end-to-end solution to solve real-life computer vision problems.


To this end, we support:

- **Supervised training**. This is the most common approach for computer vision tasks such as object detection and image classification. Supervised learning involves training a model on a labeled dataset of images. The model learns to associate specific features in the images with the corresponding labels.

- **Incremental learning**. This learning approach lets the model train on new data as it becomes available, rather than retraining the entire model on the whole dataset every time new data is added. OpenVINO™ Training Extensions supports also the class incremental approach for all tasks. In this approach, the model is first trained on a set of classes, and then incrementally updated with new classes of data, while keeping the previously learned classes' knowledge. The class incremental approach is particularly useful in situations where the number of classes is not fixed and new classes may be added over time.

.. _semi_sl_explanation:

- **Semi-supervised learning**.  This is a type of machine learning in which the model is trained on a dataset that contains a combination of labeled and unlabeled examples. The labeled examples are used to train the model, while the unlabeled examples are used to improve the model's performance by providing additional information about the underlying distribution of the data. This approach is often used when there is a limited amount of labeled data available, but a large amount of unlabeled data. This can make it more cost-effective and efficient to train models compared to traditional supervised learning, where the model is trained only on labeled data.

- **Self-supervised learning**. This is a type of machine learning where the model is trained on a dataset that contains only unlabeled examples. The model is trained to learn useful representations of the data by solving a task that can be inferred from the input itself, without human-provided labels. The objective is to learn good representations of the input data that can then be used for downstream tasks such as classification, detection, generation or clustering.


********
Contents
********


.. toctree::
   :maxdepth: 2
   :titlesonly:

   classification/index
   object_detection/index
   segmentation/index
   anomaly/index
   action/index
   visual_prompting/index
