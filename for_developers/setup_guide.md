# How to setup dev env

## Installation

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

# Install this package (Sorry the installation step is not configured yet, until that please setup PYTHONPATH env var as follows)
export PYTHONPATH=${PYTHONPATH}:${PWD}/src
```

Please see [requirements-lock.txt](requirements-lock.txt). This is what I got after the above installation steps by `pip freeze`.

## Launch training with demo recipe

```
# Please check whether your PYTHONPATH is correctly setup first

python src/otx/cli/train.py +recipe=detection/atss_r50_fpn base.data_dir=tests/assets/car_tree_bug model.otx_model.config.bbox_head.num_classes=3 trainer.max_epochs=50 trainer.check_val_every_n_epoch=10 trainer=gpu
```
