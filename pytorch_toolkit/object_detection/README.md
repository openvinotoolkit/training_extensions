# Object detection

## Pre-trained models

This repo contains scripts and tutorials for object detection models training.

### [Face Detection](face_detection.md)

Models that are able to detect faces on given images.

### [Person Vehicle Bike Detection](person_vehicle_bike_detection.md)

Models that are able to detect 3 classes of objects: person, vehicle, non-vehicle (e.g. bike).

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
