How to explain the model behavior
============================

This guide shows how to explain the model behavior, which is trained through :doc:`previous stage <how_to_train/index>`.
It allows us to show the saliency maps, which provides the locality where the model gave an attention to predict the specific category.

To be specific, this tutorial uses as an example of the ATSS model trained through ``otx train`` and saved as ``outputs/model/weights.pth``.


1. Activate the virtual environment created in the previous step.

.. code-block::

    source .otx/bin/activate

2. ``otx explain`` returns saliency maps (heatmaps with areas of focus) at the path specified by ``--save-explanation-to``.
