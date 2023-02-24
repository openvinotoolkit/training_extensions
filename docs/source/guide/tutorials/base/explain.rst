How to explain the model behavior
=================================

This guide shows how to explain the model behavior, which is trained through :doc:`previous stage <how_to_train/index>`.
It allows us to show the saliency maps, which provides the locality where the model gave an attention to predict the specific category.

To be specific, this tutorial uses as an example of the ATSS model trained through ``otx train`` and saved as ``outputs/weights.pth``.

.. note::

    This tutorial uses an object detection model for example, however for other tasks the functionality remains the same - you just need to replace the input dataset with your own.

For visualization we use images from WGISD dataset from the :doc:`object detection tutorial <how_to_train/detection>` together with trained model.

1. Activate the virtual environment 
created in the previous step.

.. code-block::

  .otx/bin/activate
  # or by this line, if you created an environment, using tox
  . venv/otx/bin/activate

2. ``otx explain`` returns saliency maps (heatmaps with red colored areas of focus) 
at the path specified by ``--save-explanation-to``.

.. code-block::

    otx explain --explain-data-roots otx-workspace-DETECTION/splitted_dataset/val/ --save-explanation-to outputs/explanation --load-weights outputs/weights.pth

3. As a result we will get a folder with a pair of generated 
images for each image in ``--explain-data-roots``: 

- saliency map - where red color means more attention of the model
- overlay - where the saliency map is combined with the original image:

.. image:: ../../../../utils/images/explain_wgisd.png
  :width: 600

