How to write OTX Configuration (recipe)
==========================================

***************
Configuration
***************

Example of ``recipe/classification/multi_class_cls``

.. code-block:: yaml

    model:
        class_path: otx.algo.classification.mobilenet_v3.MobileNetV3ForMulticlassCls
        init_args:
            label_info: 1000
            light: True

    optimizer:
        class_path: torch.optim.SGD
        init_args:
            lr: 0.0058
            momentum: 0.9
            weight_decay: 0.0001

    scheduler:
      class_path: otx.core.schedulers.LinearWarmupSchedulerCallable
      init_args:
        num_warmup_steps: 10
        main_scheduler_callable:
          class_path: lightning.pytorch.cli.ReduceLROnPlateau
          init_args:
            mode: max
            factor: 0.5
            patience: 1
            monitor: val/accuracy

    engine:
        task: MULTI_CLASS_CLS
        device: auto

    callback_monitor: val/accuracy
    data: ../../_base_/data/torchvision_base.yaml

We can use the ``~.yaml`` with the above values configured.

- ``engine``
- ``model``, ``optimizer``, ``scheduler``
- ``data``
- ``callback_monitor``

The basic configuration is the same as the configuration configuration format for jsonargparse.
`Jsonargparse Documentation <https://jsonargparse.readthedocs.io/en/v4.27.4/#configuration-files>_`


***********************
Configuration Overrides
***********************

Here we provide a feature called ``overrides``.
This feature allows you to override the values need from the default configuration.
Currently, you can overrides ``data``, ``callbacks``, ``logger``, and other single value configurations.
Also, you can use ``reset`` to reset the configurations from the default values to the new values.

To update single value configurations, just set them in the overrides.

.. code-block:: yaml

    ...
    overrides:
      max_epochs: 10 # update to 10
      data:
        image_color_channel: BGR # update to BGR
    ...

If you want to add new configuration which isn't set before, just set them in the overrides and it will be appended.

.. code-block:: yaml

    ...
    overrides:
      new_configuration: 1 # add new_configuration with 1
    ...

.. note::

  You can see the final configuration with the command below.

  .. code-block:: shell

      $ otx train --config <config-file-path> --print_config


--------------
Data overrides
--------------

``data`` can currently be provided as a list of different transforms.
The way to override this is as follows.

Let's try to change the size of Resize and the prob of RandomFlip which are already set in `base data configuration of instance segmentation <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/_base_/data/instance_segmentation.yaml>`_.
To change them, you can just set the values in the overrides.

.. code-block:: yaml

    ...
    overrides:
      data:
        train_subset:
          transforms:
            - class_path: otx.core.data.transform_libs.torchvision.Resize
              init_args:
                size: # update `size` from 1024 to 512
                  - 512
                  - 512
            # Pad is used as is because it is not set here
            - class_path: otx.core.data.transform_libs.torchvision.RandomFlip
              init_args:
                prob: 0 # update `prob` from 0.5 to 0
            # ToDtype and Normalize are used as is because they are not set here
    ...

Like single value configurations, when adding new transforms in overrides it will be appended.

.. code-block:: yaml

    ...
    overrides:
      data:
        train_subset:
          transforms:
            - class_path: new_transform
    ...


----------------------------
Callbacks & Logger overrides
----------------------------

``callbacks`` and ``logger`` can currently be provided as a list of different callbacks and loggers.
The way to override this is as follows.

For example, if you want to change the patience of EarlyStopping, you can configure the overrides like this

.. code-block:: yaml

    overrides:
    ...
        callbacks:
          - class_path: ligthning.pytorch.callbacks.EarlyStopping
            init_args:
              patience: 3


---------------
Reset overrides
---------------

If you want to **reset** the configurations to the default values, especially ``data``, ``callbacks``, or ``logger`` that are difficult to be reset, you can use the ``reset`` keyword.
The way to override this is as follows.

Let's try to reset all transforms which are already set in `base data configuration of instance segmentation <https://github.com/openvinotoolkit/training_extensions/blob/develop/src/otx/recipe/_base_/data/instance_segmentation.yaml>`_.
To reset them, you can just add the keys in ``reset`` in the overrides.
``reset`` also supports both types, string and list.
If you want to reset single one, string or list can be used.
But if you want to reset multiple ones, list should be used.

.. tab-set::

  .. tab-item:: single one

    .. code-block:: yaml

        ...
        overrides:
          reset:
            - data.train_subset.transforms
          # or
          # reset: data.train_subset.transforms
          data:
            train_subset:
              transforms:
                # previous ones are not used anymore
                - class_path: new_transform_1
                - class_path: new_transform_2
        ...

  .. tab-item:: multiple ones

    .. code-block:: yaml

        ...
        overrides:
          reset:
            - data.train_subset.transforms
            - data.val_subset.transforms
            - data.test_subset.transforms
            - callbacks
          # reset: data.train_subset.transforms cannot be used because there are multiple resets
          data:
            train_subset:
              transforms:
                # previous ones are not used anymore
                - class_path: new_transform_1
                - class_path: new_transform_2
            val_subset:
              transforms:
                # previous ones are not used anymore
                - class_path: new_transform_1
                - class_path: new_transform_2
            test_subset:
              transforms:
                # previous ones are not used anymore
                - class_path: new_transform_1
                - class_path: new_transform_2
          ...
          callbacks:
            # previous ones are not used anymore
            - class_path: new_callback_1
            - class_path: new_callback_2
        ...
