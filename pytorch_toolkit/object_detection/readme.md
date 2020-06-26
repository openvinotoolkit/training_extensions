# Object detection

## Pre-trained models

This repo contains scripts and tutorials for object detection models training.

* [Face Detection](face-detection/readme.md) - models that are able to detect faces on given images.
* [Person Vehicle Bike Detection](person-vehicle-bike-detection/readme.md) - models that are able to detect 3 classes of objects: person, vehicle, non-vehicle (e.g. bike).
* [Horizontal Text Detection](horizontal-text-detection/readme.md) - model that is able to detect more or less horizontal text with high speed.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* >=3.6
* PyTorch\* 1.4.0
* OpenVINO™ 2020.2 with Python API
* mmdetection (../../external/mmdetection)

### Installation

1. Create virtual environment and build mmdetection:
```bash
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```
