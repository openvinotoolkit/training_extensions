Multi-GPU Support
=================

Overview
--------

OpenVINO™ Training Extensions now supports operations in a multi-GPU environment, offering faster computation speeds and enhanced performance. With this new feature, users can efficiently process large datasets and complex models, significantly reducing the time required for machine learning and deep learning tasks.

Benefits of Multi-GPU Support
-----------------------------

- **Speed Improvement**: Training times can be greatly reduced by utilizing multiple GPUs in parallel.
- **Large Dataset Handling**: Load larger datasets into memory and work with larger batch sizes.
- **Efficient Resource Utilization**: Maximize the computational efficiency by fully utilizing the GPU resources of the system.

How to Set Up Multi-GPU
-----------------------

Setting up multi-GPU in OpenVINO™ Training Extensions is straightforward. Follow these steps to complete the setup:

1. **Environment Check**: Ensure that multiple GPUs are installed in your system and that all GPUs are compatible with OpenVINO™ Training Extensions.
2. **Driver Installation**: Install the latest GPU drivers to ensure all GPUs are properly recognized and available for use.
3. **Configuration**: Activate the multi-GPU option in the OpenVINO™ Training Extensions configuration file or through the user interface.

Using Multi-GPU
---------------

Once the multi-GPU feature is enabled, you can use multi-GPU for model training as follows:

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train \
                        ... \
                        --engine.num_devices 2

    .. tab-item:: API

        .. code-block:: python

            from otx.engine import Engine

            engine = Engine.from_config(
                        ...
                        num_devices=2,
                    )

            engine.train(...)
