# How to setup dev env

## Installation

### With Conda

```console
# Create venv from conda
conda create -n otx-v2 python=3.11
conda activate otx-v2

# Install PyTorch and TorchVision
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependency
pip install lightning datumaro omegaconf hydra-core

# Install mmcv (mmdet)
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet

# Install this package
pip install -e .
```

### With PIP & 'otx install'

```console
# Create venv
python -m venv venv
source venv/bin/activate

# Install this package
pip install -e .

# OTX --help
otx --help

# Install torch & lightning base on user environments
otx install
# or 'otx install -v' (Verbose mode)

# Install other mmlab library or optional-dependencies
otx install --option dev
# or 'otx install --option mmpretrain'
```

Please see [requirements-lock.txt](requirements-lock.txt). This is what I got after the above installation steps by `pip freeze`.

## Launch training with demo template

- Launch detection task ATSS-R50-FPN template

```console
otx train +recipe=detection/atss_r50_fpn base.data_dir=tests/assets/car_tree_bug model.otx_model.config.bbox_head.num_classes=3 trainer.max_epochs=50 trainer.check_val_every_n_epoch=10 trainer=gpu base.work_dir=outputs/test_work_dir base.output_dir=outputs/test_output_dir
```

- Change subset names, e.g., "train" -> "train_16" (for training)

```console
otx train ... data.train_subset_name=<arbitrary-name> data.val_subset_name=<arbitrary-name> data.test_subset_name=<arbitrary-name>
```

- Do test with the best validation model checkpoint

```console
otx train ... test=true
```
