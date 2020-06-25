# Object Re-Identification

## Pre-trained models

This repo contains scripts and tutorials for object reidentification models training.

* [Person Re-Identification](person_reidentification)
* [Vehicle Re-Identification](vehicle_reidentification)
* [Face recognition](face_recognition)

## Setup

### Prerequisites

* Ubuntu\* >=16.04
* Python\* >=3.6
* PyTorch\* 1.4.0
* OpenVINO™ 2020.2 with Python API
* [deep-object-reid](https://github.com/opencv/deep-object-reid)

### Installation

1. Create virtual environment:
```bash
cd $(git rev-parse --show-toplevel)/pytorch_toolkit/object_reidentification
bash init_venv.sh
```

2. Activate virtual environment:
```bash
. venv/bin/activate
```
