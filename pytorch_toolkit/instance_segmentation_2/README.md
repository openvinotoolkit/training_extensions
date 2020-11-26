# Instance Segmentation

## Pre-trained models

This repo contains scripts and tutorials for instance segmentation models training.

* [COCO instance segmentation](model_templates/coco-instance-segmentation/readme.md) - models that are able to instantiate segmentation.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* >=3.6
* PyTorch\* 1.4.0
* OpenVINO™ 2020.4 or later with Python API
* mmdetection (../../external/mmdetection)

### Installation

1. Create virtual environment and build mmdetection:
```bash
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate```
