# Speech To Text

This is a Speech To Text toolbox, that is part of [OpenVINO™ Training Extensions](https://github.com/opencv/openvino_training_extensions).

### Installation

1. Create virtual environment:
```bash
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```

## Training and Evaluation using python API

To train and evaluate net:

```bash
python3 pl_train_quartznet.py \
    --cfg configs/quartznet5x5.json \
    --gpus <n_gpus>
```

### Convert to OpenVino

```bash
python3 pl_train_quartznet.py \
    --cfg configs/quartznet5x5.json \
    --export
```
