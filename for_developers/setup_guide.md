# How to setup dev env

## Installation

### Requirements

CUDA >= 11.8
torchvision >= 0.16
python >= 3.9
pip >= 23.3.2

### With Conda

```console
# Create venv from conda
conda create -n otx-v2 python=3.11
conda activate otx-v2

# Install PyTorch and TorchVision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install otx with core requirements
pip install -e .

# otx install (install mmX)
otx install -v
```

In case of the error: `File "setup.py" not found. Directory cannot be installed in editable mode:`, uprade pip:

```
pip install --upgrade pip
```

### With PIP & 'otx install'

```console
# Create venv
python3.10 -m venv venv
source venv/bin/activate

# Install this package
pip install -e .

# OTX --help
otx --help

# Upgrade pip
pip install --upgrade pip

# Install torch & lightning base on user environments
otx install -v
# or 'otx install' (Not verbose mode)
```

Please see [requirements-lock.txt](requirements-lock.txt). This is what I got after the above installation steps by `pip freeze`.

## Launch training with demo template

- Launch detection task ATSS-R50-FPN template

  ```console
  otx train --config src/otx/recipe/detection/atss_r50_fpn.yaml --data_root tests/assets/car_tree_bug --model.num_classes=3 --max_epochs=50 --check_val_every_n_epoch=10 --engine.device gpu --engine.work_dir ./otx-workspace
  ```

- Change subset names, e.g., "train" -> "train_16" (for training)

  ```console
  otx train ... --data.config.train_subset.subset_name <arbitrary-name> --data.config.val_subset.subset_name <arbitrary-name> --data.config.test_subset.subset_name <arbitrary-name>
  ```

- Do train with the existing model checkpoint for resume

  ```console
  otx train ... --checkpoint <checkpoint-path>
  ```

- Do experiment with deterministic operations and the fixed seed

  ```console
  otx train ... --deterministic True --seed <arbitrary-seed>
  ```

- Do test with the existing model checkpoint

  ```console
  otx test ... --checkpoint=<checkpoint-path>
  ```

  `--deterministic True` might affect to the model performance. Please see [this link](https://lightning.ai/docs/pytorch/stable/common/trainer.html#deterministic). Therefore, it is not recommended to turn on this option for the model performance comparison.
