Action Detection
================

Sptio-Temporal action detection is the problem of localizing the actor(spatial detection) and action(temporal detection). We solve this problem by combining 3D action classification backbone and 2D object detection model. We can combine these two models in several ways. Currently, we support the simplest way. The other ways will be supported in near future.

X3D + Fast-RCNN architecture comes from `X3D paper <https://arxiv.org/abs/2004.04730>`_. This model requires pre-computed actor proposals. Actor pre-proposals can be obtained from `COCO <https://cocodataset.org/#home>`_ pre-trained 2D object detector (e.g. `Faster-RCNN <https://arxiv.org/abs/1506.01497>`_, `ATSS <https://arxiv.org/abs/1912.02424>`_). If the custom dataset requires finetuning of 2d object detector, please refer :doc:`otx.algorithms.detection <../object_detection/object_detection>`. Region-of-interest (RoI) features are extracted at the last feature map of X3D by extending a 2D proposal at a keyframe into a 3D RoI by replicating it along the temporal axis. The RoI features fed into the roi head of Fast-RCNN.

For better transfer learning we use the following algorithm components:

- ``Augmentations``: We use only random crop and random flip for the training pipeline

- ``Optimizer``: We use `SGD <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ optimizer with the weight decay set to **1e-4** and momentum set to **0.9**.

- ``Loss functions``: For the multi-label case binary cross entropy loss is used. In the other case, `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_ is used for the categories classification.

**************
Dataset Format
**************

For the dataset handling inside OpenVINO™ Training Extensions, we use `Dataset Management Framework (Datumaro) <https://github.com/openvinotoolkit/datumaro>`_. Since current Datumaro does not support `AVA dataset <http://research.google.com/ava/>`_ format, therefore conversion to `CVAT dataset format <https://opencv.github.io/cvat/docs/manual/advanced/xml_format/>`_ is needed. Currently, we offer conversion code from the AVA dataset format to the CVAT dataset format. Please refer
`this script <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/algorithms/action/utils/convert_public_data_to_cvat.py>`_


******
Models
******

We support the following ready-to-use model templates for transfer learning:

+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+---------------------+-------------------------+
| Template ID                                                                                                                                                                             | Name          | Complexity (GFLOPs) | Model size (MB)         |
+=========================================================================================================================================================================================+===============+=====================+=========================+
| `Custom_Action_Detection_X3D_FAST_RCNN <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/action/action_detection/x3d_fast_rcnn.yaml>`_                | x3d_fast_rcnn | 13.04               | 8.32                    |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+---------------------+-------------------------+

To see which models are available for the task, the following command can be executed:

.. code-block:: shell

        (otx) ...$ otx find --task ACTION_DETECTION

In the table below the **mAP** on some academic datasets are presented. Each model is trained using `Kinetics-400 <https://www.deepmind.com/open-source/kinetics>`_ pre-trained weight with single Nvidia GeForce RTX3090.

+----------------+-------+-----------+
| Model name     | JHMDB | UCF101-24 |
+================+=======+===========+
| x3d_fast_rcnn  | 92.14 |   80.7    |
+----------------+-------+-----------+
