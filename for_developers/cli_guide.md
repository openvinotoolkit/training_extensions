# How to use OTX CLI

## Installation

Please see [setup_guide.md](setup_guide.md).

## otx help

```console
otx --help
```

```powershell
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────╮
│ Usage: otx [-h] [-v] {install,train,test,predict,export} ...                                    │
│                                                                                                 │
│                                                                                                 │
│ OpenVINO Training-Extension command line tool                                                   │
│                                                                                                 │
│                                                                                                 │
│ Options:                                                                                        │
│   -h, --help            Show this help message and exit.                                        │
│   -v, --version         Display OTX version number.                                             │
│                                                                                                 │
│ Subcommands:                                                                                    │
│   For more details of each subcommand, add it as an argument followed by --help.                │
│                                                                                                 │
│                                                                                                 │
│   Available subcommands:                                                                        │
│     install             Install OTX requirements.                                               │
│     train               Trains the model using the provided LightningModule and OTXDataModule.  │
│     test                Run the testing phase of the engine.                                    │
│     predict             Run predictions using the specified model and data.                     │
│     export              Export the trained model to OpenVINO Intermediate Representation (IR) o │
│                         ONNX formats.                                                           │
│                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The subcommand can get help output in the following way.
For basic subcommand help, the Verbosity Level is 0. In this case, the CLI provides a Quick-Guide in markdown.

```console
# otx {subcommand} --help
otx train --help
```

```powershell
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                OpenVINO™ Training Extensions CLI Guide                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Github Repository:
https://github.com/openvinotoolkit/training_extensions.

A better guide is provided by the documentation.
╭─ Quick-Start ─────────────────────────────────────────────────────────╮
│                                                                       │
│  1 you can train with data_root only. then OTX will provide default   │
│    model.                                                             │
│                                                                       │
│                                                                       │
│  otx train --data_root <DATASET_PATH>                                 │
│                                                                       │
│                                                                       │
│  2 you can pick a model or datamodule as Config file or Class.        │
│                                                                       │
│                                                                       │
│  otx train                                                            │
│  --data_root <DATASET_PATH>                                           │
│  --model <CONFIG | CLASS_PATH_OR_NAME> --data <CONFIG |               │
│  CLASS_PATH_OR_NAME>                                                  │
│                                                                       │
│                                                                       │
│  3 Of course, you can override the various values with commands.      │
│                                                                       │
│                                                                       │
│  otx train                                                            │
│  --data_root <DATASET_PATH>                                           │
│  --max_epochs <EPOCHS, int> --checkpoint <CKPT_PATH, str>             │
│                                                                       │
│                                                                       │
│  4 If you have a complete configuration file, run it like this.       │
│                                                                       │
│                                                                       │
│  otx train --data_root <DATASET_PATH> --config <CONFIG_PATH, str>     │
│                                                                       │
│                                                                       │
│ To get more overridable argument information, run the command below.  │
│                                                                       │
│                                                                       │
│  # Verbosity Level 1                                                  │
│  otx train [optional_arguments] -h -v                                 │
│  # Verbosity Level 2                                                  │
│  otx train [optional_arguments] -h -vv                                │
│                                                                       │
╰───────────────────────────────────────────────────────────────────────╯
```

For Verbosity Level 1, it shows Quick-Guide & the essential arguments.

```console
otx train --help -v
```

```powershell
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                OpenVINO™ Training Extensions CLI Guide                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Github Repository:
https://github.com/openvinotoolkit/training_extensions.

A better guide is provided by the documentation.
╭─ Quick-Start ─────────────────────────────────────────────────────────╮
│  ...                                                                  │
╰───────────────────────────────────────────────────────────────────────╯
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────╮
│ Usage: otx [options] train [-h] [-c CONFIG] [--print_config [=flags]]                           │
│                            [--data_root DATA_ROOT] [--task TASK]                                │
│                            [--engine CONFIG]                                                    │
│                            [--engine.work_dir WORK_DIR]                                         │
│                            [--engine.checkpoint CHECKPOINT]                                     │
│                            [--engine.device {auto,gpu,cpu,tpu,ipu,hpu,mps}]                     │
│                            [--model.help CLASS_PATH_OR_NAME]                                    │
│                            [--model CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]         │
│                            [--data CONFIG]                                                      │
│                            [--optimizer CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]     │
│                            [--scheduler CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE]     │
│                                                                                                 │
...
```

For Verbosity Level 2, it shows all available arguments.

```console
otx train --help -vv
```

## otx {subcommand} --print_config

Preview all configuration values that will be executed through that command line.

```console
otx train --config <config-file-path> --print_config
```

```yaml
data_root: tests/assets/car_tree_bug
callback_monitor: val/map_50
engine:
  task: DETECTION
  work_dir: ./otx-workspace
  device: auto
model:
  class_path: otx.algo.detection.atss.ATSS
  init_args:
    num_classes: 1000
    variant: mobilenetv2
optimizer: ...
scheduler: ...
data:
  task: DETECTION
  config:
    data_format: coco_instances
    train_subset: ...
    val_subset: ...
    test_subset: ...
    mem_cache_size: 1GB
    mem_cache_img_max_size: null
    image_color_channel: RGB
    include_polygons: false
max_epochs: 2
deterministic: false
precision: 16
callbacks: ...
logger: ...
```

Users can also pre-generate a config file with an example like the one below.

```console
otx train --config <config-file-path> --print_config > config.yaml
```

## otx {subcommand}

Use Auto-Configuration

```console
otx train --data_root <dataset-root> --task <TASK>
```

Use Configuration file

```console
otx train --config <config-file-path> --data_root <dataset-root>
```

Override Parameters

```console
otx train ... --model.num_classes <num-classes> --max_epochs <max-epochs>
```

Testing with checkpoint

```console
otx test ... --checkpoint <checkpoint-path>
```

Export to OpenVINO IR model or ONNX (Default="OPENVINO")

```console
otx export ... --checkpoint <checkpoint-path> --export_format <export-format>
```

Testing with Exported model output

```console
otx test ... --checkpoint <checkpoint-path-IR-xml-or-onnx>
```

## How to write OTX Configuration (recipe)

### Configuration

Example of `recipe/classification/multi_class_cls`

```yaml
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
```

We can use the `~.yaml` with the above values configured.

- `engine`
- `model`, `optimizer`, `scheduler`
- `data`
- `callback_monitor`
  The basic configuration is the same as the configuration configuration format for jsonargparse.
  [Jsonargparse Documentation](https://jsonargparse.readthedocs.io/en/v4.27.4/#configuration-files)

### Configuration overrides

Here we provide a feature called `overrides`.

```yaml
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
```

This feature allows you to override the values need from the default configuration.
You can see the final configuration with the command below.

```console
otx train --config <config-file-path> --print_config
```

### Callbacks & Logger overrides

`callbacks` and `logger` can currently be provided as a list of different callbacks and loggers. The way to override this is as follows.

For example, if you want to change the patience of EarlyStopping, you can configure the overrides like this

```yaml
...

overrides:
  ...
  callbacks:
    - class_path: ligthning.pytorch.callbacks.EarlyStopping
      init_args:
        patience: 3
```
