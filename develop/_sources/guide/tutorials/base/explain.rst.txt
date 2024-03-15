How to explain the model behavior
=================================

This guide explains the model behavior, which is trained through :doc:`previous stage <how_to_train/index>`.
It allows displaying the saliency maps, which provide the locality where the model gave an attention to predict a specific category.

To be specific, this tutorial uses as an example of the ATSS model trained through ``otx train`` and saved as ``otx-workspace/.latest/train/checkpoints/epoch_*.pth``.

.. note::

    This tutorial uses an object detection model for example, however for other tasks the functionality remains the same - you just need to replace the input dataset with your own.

For visualization we use images from WGISD dataset from the :doc:`object detection tutorial <how_to_train/detection>` together with trained model.

1. Activate the virtual environment 
created in the previous step.

.. code-block:: shell

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

2. ``otx explain`` returns saliency maps (heatmaps with red colored areas of focus)

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            Need to update!

    .. tab-item:: API

        .. code-block:: python

            Need to update!


3. To specify the algorithm of saliency map creation for classification, 
we can define the ``--explain-algorithm`` parameter.

- ``activationmap`` - for activation map classification algorithm 
- ``eigencam`` -  for Eigen-Cam classification algorithm
- ``classwisesaliencymap`` -  for Recipro-CAM classification algorithm, this is a default method

For detection task, we can choose between the following methods:

- ``activationmap`` - for activation map detection algorithm
- ``classwisesaliencymap`` - for DetClassProbabilityMap algorithm (works for single-stage detectors only)

.. note::

  Learn more about Explainable AI and its algorithms in :doc:`XAI explanation section <../../explanation/additional_features/xai>`


4. As a result we will get a folder with a pair of generated 
images for each image in ``--input``: 

- saliency map - where red color means more attention of the model
- overlay - where the saliency map is combined with the original image:

.. image:: ../../../../utils/images/explain_wgisd.png
  :width: 600

