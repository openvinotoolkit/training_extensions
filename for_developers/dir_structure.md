```console
root/
    algo/                       # Custom algo (e.g., hierarchical_cls_head)
    cli/                        # CLI entrypoints
    engine/                     # OTX Engine with Entry Point
    core/
        config/                 # Structured data type object for configurations
        data/                   # Data related things
            dataset/            # OTXDataset
                base.py
                detection.py
                ...
            entity/             # OTXDataEntity
                base.py
                detection.py
                ...
            transform_libs/     # To support transform libraries (e.g., torchvision)
            factory.py          # Factory to instantiate data related objects
            module.py           # OTXDataModule
        model/                  # Model related things
            entity/             # OTXModel
                base.py
                detection.py
                ...
            module/             # OTXLitModule
                base.py
                detection.py
                ...
        types/                  # Enum definitions (e.g. OTXTaskType)
        utils/                  # Utility functions
    recipe/                     # Recipe YAML config for each model we support
        _base_/                 # Default YAML config files
        detection/              # (e.g., rtmdet_tiny)
        ...
    tools/                      # Python runnable scripts for some TBD use cases
```
