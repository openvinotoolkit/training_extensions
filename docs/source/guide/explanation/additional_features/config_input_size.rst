Configurable Input Size
=======================

This feature makes OTX use smaller input resolutions to expedite training and inference times,
or opt for larger input size to enhance overall model capabilities.

Rather than manually modifying values within the data pipeline, a streamlined approach is provided.
You can change the model's input size by simply adding an argument during `train`, `eval`, or `export`.
Furthermore, when using a model weight trained on an input size other than the default,
OTX automatically aligns data pipelines to input size during `eval`` and `export` processes.

You can use this feature using the following command:

.. code-block::

    $ otx train \
          ... \
          params --learning_parameters.input_size input_size

The available input sizes are currently as follows:

- 64x64 (exclusively for classification)
- 128x128 (exclusively for classification)
- 256x256
- 384x384
- 512x512
- 1024x1024

.. Note::
    Using smaller input size with datasets having lower image resolutions or larger objects can yield a speed advantage with minimal impact on model performance.
    But note that the choice of small input size can potentially influence model performance based on the task, model architecture, and dataset characteristics.
