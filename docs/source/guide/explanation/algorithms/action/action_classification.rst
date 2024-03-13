Action Classification
=====================

Action classification is a problem of identifying the action that is being performed in a video. The input to the algorithm is a sequence of video frames, and the output is a label indicating the action that is being performed.

For supervised learning we use the following algorithms components:

- ``Augmentations``: We use standard data augmentations for videos, including random resizing and random cropping, horizontal flipping. We randomly sample a segment of frames from each video during training.

- ``Optimizer``: We use the Adam with weight decay fix (AdamW) optimizer.

- ``Learning rate schedule``: We use a step learning rate schedule, where the learning rate is reduced by a factor of 10 after a fixed number of epochs. We also use the Linear Warmup technique to gradually increase the learning rate at the beginning of training.

- ``Loss function``: We use the Cross-Entropy Loss as the loss function. 

**************
Dataset Format
**************

We support the popular action classification formats, such as `Jester <https://developer.qualcomm.com/software/ai-datasets/jester>`_, `HMDB51 <https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_, `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_. Specifically, these formats will be converted into our `internal representation <https://github.com/openvinotoolkit/training_extensions/tree/develop/tests/assets/cvat_dataset/action_classification/train>`_ using the `Datumaro <https://github.com/openvinotoolkit/datumaro>`_ dataset handler.

The names of the annotations files and the overall dataset structure should be the same as the original dataset.

Refer to our tutorial for more information on how to train, validate, and optimize action classification models.

******
Models
******

Currently OpenVINO™ Training Extensions supports `X3D <https://arxiv.org/abs/2004.04730>`_ and `MoViNet <https://arxiv.org/pdf/2103.11511.pdf>`_ for action classification.

+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+---------------------+-------------------------+
| Template ID                                                                                                                                                                            | Name    | Complexity (GFLOPs) | Model size (MB)         |
+========================================================================================================================================================================================+=========+=====================+=========================+
| `Custom_Action_Classification_X3D <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/action/action_classification/x3d.yaml>`_                         | X3D     | 2.49                | 3.79                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+---------------------+-------------------------+
| `Custom_Action_Classificaiton_MoViNet <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/action/action_classification/movinet.yaml>`_                 | MoViNet | 2.71                | 3.10                    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+---------------------+-------------------------+

To see which models are available for the task, the following command can be executed:

.. code-block:: shell

        (otx) ...$ otx find --task ACTION_CLASSIFICATION

In the table below the **top-1 accuracy** on some academic datasets are presented. Each model is trained with single NVIDIA GeForce RTX 3090.

+-----------------------+------------+-----------------+
| Model name            | HMDB51     | UCF101          |
+=======================+============+=================+
| X3D                   | 67.19      | 87.89           |
+-----------------------+------------+-----------------+
| MoViNet               | 62.74      | 81.32           |
+-----------------------+------------+-----------------+
