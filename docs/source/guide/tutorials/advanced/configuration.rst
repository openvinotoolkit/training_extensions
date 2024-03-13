How to write OTX Configuration (recipe)
==========================================

***************
Configuration
***************

Example of ``recipe/classification/multi_class_cls``

.. code-block:: yaml

    model:
        class_path: otx.algo.classification.mobilenet_v3_large.MobileNetV3ForMulticlassCls
        init_args:
            num_classes: 1000
            light: True

    optimizer:
        class_path: torch.optim.SGD
        init_args:
            lr: 0.0058
            momentum: 0.9
            weight_decay: 0.0001

    scheduler:
        class_path: otx.algo.schedulers.WarmupReduceLROnPlateau
        init_args:
            warmup_steps: 10
            mode: max
            factor: 0.5
            patience: 1
            monitor: val/accuracy

    engine:
        task: MULTI_CLASS_CLS
        device: auto

    callback_monitor: val/accuracy
    data: ../../_base_/data/mmpretrain_base.yaml

We can use the ``~.yaml`` with the above values configured.

- ``engine``
- ``model``, ``optimizer``, ``scheduler``
- ``data``
- ``callback_monitor``

The basic configuration is the same as the configuration configuration format for jsonargparse.
`Jsonargparse Documentation <https://jsonargparse.readthedocs.io/en/v4.27.4/#configuration-files>_`

### Configuration overrides

Here we provide a feature called ``overrides``.

.. code-block:: yaml

    ...

    overrides:
        data:
            config:
            train_subset:
                transforms:
                - type: LoadImageFromFile
                - backend: cv2
                    scale: 224
                    type: RandomResizedCrop
                - direction: horizontal
                    prob: 0.5
                    type: RandomFlip
                - type: PackInputs
    ...

This feature allows you to override the values need from the default configuration.
You can see the final configuration with the command below.

.. code-block:: shell

    $ otx train --config <config-file-path> --print_config

### Callbacks & Logger overrides

``callbacks`` and ``logger`` can currently be provided as a list of different callbacks and loggers. The way to override this is as follows.

For example, if you want to change the patience of EarlyStopping, you can configure the overrides like this

.. code-block:: yaml

    overrides:
    ...
        callbacks:
            - class_path: ligthning.pytorch.callbacks.EarlyStopping
            init_args:
                patience: 3
