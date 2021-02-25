# Object detection

## Pre-trained models

This repo contains scripts and tutorials for object detection models training.

* [Face Detection](model_templates/face-detection/readme.md) - models that are able to detect faces.
* [Horizontal Text Detection](model_templates/horizontal-text-detection/readme.md) - model that is able to detect more or less horizontal text with high speed.
* [Person Detection](model_templates/person-detection/readme.md) - models that are able to detect persons.
* [Person Vehicle Bike Detection](model_templates/person-vehicle-bike-detection/readme.md) - models that are able to detect 3 classes of objects: person, vehicle, non-vehicle (e.g. bike).
* [Vehicle Detection](model_templates/vehicle-detection/readme.md) - models that are able to detect vehicles.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* >=3.6
* PyTorch\* 1.5.1
* OpenVINO™ 2020.4 with Python API
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
