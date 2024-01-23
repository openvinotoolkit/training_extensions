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

Use Configuration file

```console
otx train --config <config-file-path> --data_root <dataset-root>
```

Override Parameters

```console
otx train ... --model.num_classes <num-classes> --max_epochs <max-epochs>
```
