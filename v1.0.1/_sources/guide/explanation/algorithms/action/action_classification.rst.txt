Action Classification
==================

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

We support `X3D <https://arxiv.org/abs/2004.04730>`_ for action classification. X3D is a deep learning model that was proposed in the paper "X3D: Expanding Architectures for Efficient Video Recognition" by Christoph Feichtenhofer. The model is an extension of the popular 2D convolutional neural network (CNN) architectures to the 3D domain, allowing it to efficiently process spatiotemporal information in videos. 

Currenly OpenVINO™ Training Extensions supports X3D-S model with below template:

+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+---------------------+-------------------------+
| Template ID                                                                                                                                                                   | Name    | Complexity (GFLOPs) | Model size (MB)         |
+===============================================================================================================================================================================+=========+=====================+=========================+
| `Custom_Action_Classification_X3D <https://github.com/openvinotoolkit/training_extensions/blob/develop/otx/algorithms/action/configs/classification/x3d/template.yaml>`_      | X3D     | 2.49                | 3.79                    |
+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------+---------------------+-------------------------+


In the table below the **top-1 accuracy** on some academic datasets are presented. Each model is trained with single Nvidia GeForce RTX3090.

+-----------------------+------------+-----------------+
| Model name            | HMDB51     | UCF101          |
+=======================+============+=================+
| X3D                   | 67.19      | 87.89           |
+-----------------------+------------+-----------------+
