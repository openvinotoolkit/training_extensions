Diffusion
=================

Diffusion models are generative models that learn to generate data by gradually adding noise to the input and then learning to reverse this process. This approach enables them to capture complex patterns and distributions in the data, making them suitable for a variety of tasks, including image generation, text generation, and audio synthesis. However, training diffusion models can be computationally expensive and requires large amounts of data. Additionally, the quality of the generated data depends heavily on the quality of the training data and the hyperparameters used during training.

This section examines the solutions for diffusion offered by the OpenVINO Training Extensions library.


Dataset Format
**************
At the moment, the diffusion task supports the COCO captions dataset format:

.. code-block::

└─ Dataset/
    ├── dataset_meta.json # a list of custom labels (optional)
    ├── images/
    │   ├── train/
    │   │   ├── <image_name1.ext>
    │   │   ├── <image_name2.ext>
    │   │   └── ...
    │   └── val/
    │       ├── <image_name1.ext>
    │       ├── <image_name2.ext>
    │       └── ...
    └── annotations/
        ├── <task>_<subset_name>.json
        └── ...


Models
******
As mentioned above, the goal of diffusion is to learn a generative model that can progressively transform a random noise vector into a realistic sample from a given data distribution. This process involves adding noise to the input data in a controlled manner and then training a model to reverse this process, gradually refining the noise into a meaningful output. Diffusion models are particularly effective at capturing complex patterns and dependencies in the data, making them suitable for a wide range of generative tasks. OpenVINO Training Extensions supports Stable Diffusion pipeline v1.4 that consists of 3 models:
 - text encoder (CLIP ViT-L/14),
 - autoencoder
 - diffusion model (UNet)
Pipeline is based on HuggingFace implementation pre-trained on LAION-5B dataset. In OpenVINO Training Extensions, we use the fine-tuning approach to train the model on the target dataset.



Training Parameters
~~~~~~~~~~~~~~~~~~~~

The following parameters can be changed during training:
- ``Loss``: Loss is computed as the mean squared error between target noise and predicted noise. The default loss is ``MSE`` and cannot be changed.
- ``Optimizer``: The default optimizer is ``AdamW`` and cannot be changed. It uses the following parameters that can be changed:
   - ``Learning Rate``: The default learning rate is ``0.00001``.
   - ``Betas``: The default betas are is ``[0.9, 0.999]``.
   - ``Weight Decay``: The default weight decay is ``0.01``.
   - ``Epsilon``: The default epsilon is ``1e-8``.
