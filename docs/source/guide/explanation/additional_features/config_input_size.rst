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

- 64x64 (only for classification)
- 128x128 (only for classification)
- 224x224 (only for classification)
- 256x256
- 384x384
- 512x512
- 768x768
- 1024x1024
- Default (per-model default input size)
- Auto (adaptive to dataset statistics)

.. _adaptive-input-size:

Adaptive Input Size
-------------------

"Auto" mode tries to automatically select the right size
based on given dataset statictics.

1. OTX analyzes the input dataset to get robust statistics.

2. Input size is initially set to typical large image size.

.. code-block::

    input_size = large_image_size

3. (Optionally) Input size is adjusted by object sizes in the dataset, if any.
   The input size from image size is rescaled accoridng to the ratio of
   minimum recongnizable object size of models, which is typically 16x16 ~ 32x32,
   and the typical small object size in the dataset.
   In short, if objects are 64x64 in general in 512x512 image,
   it will be down-scaled to 256x256 as 32x32 objects are enough to be detected.

.. code-block::

    input_size = input_size * MIN_RECOGNIZABLE_OBJECT_SIZE / small_object_size

4. Select the closest size from standard preset sizes

5. Restrict scale-up

.. code-block::

    input_size = min(input_size, default_model_input_size)


.. Note::
    Using smaller input size with datasets having lower image resolutions or larger objects can yield a speed advantage with minimal impact on model performance.
    But note that the choice of small input size can potentially influence model performance based on the task, model architecture, and dataset characteristics.
