# Object detection models

This repository contains training scripts for object detection models.

## [Face Detection](face_detection.md)

This repository contains training scripts for the lightweight SSD-based face detector. The detector is based on the MobileNetV2 backbone and has a single SSD head with manually designed anchors. As a result, it has computational complexity 0.51 GMACs and 1.03 M of parameters.

## [Person Vehicle Bike Detection](person_vehicle_bike_detection.md)

The detection network model provides detection of 3 class objects: vehicle, pedestrian, non-vehicle (ex: bikes).
This detector was trained on the data from crossroad cameras.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.7
* PyTorch\* 1.0.1
* OpenVINO™ 2020.1 with Python API

### Installation

1. Create virtual environment and build mmdetection:
```bash
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```
