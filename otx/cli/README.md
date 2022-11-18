# OTX CLI

**O**penVINO **T**raining **E**xtensions **C**ommand **L**ine **I**nterface (OTX CLI), is the `otx` tool that contains set of commands needed to operate with deep learning models. Also there is an example how to work with deep learning models direclty from Python using OTX Task interfaces.

## OTX CLI contents

### OTX Commands

- `otx find` - search for model templates.
- `otx train` - run training of a particular model template.
- `otx optimize` - run optimization of trained model.
- `otx eval` - run evaluation of trained model on a particular dataset.
- `otx explain` - run model's explanation and save saliency map for feature vector to the dump path.
- `otx export` - export trained model to the OpenVINO format in order to efficiently run it on Intel hardware.
- `otx demo` - run model inference on images, videos, webcam in order to see how it works on user's data.
- `otx deploy` - create openvino.zip with self-contained python package, demo application and exported model.

### Jupyter notebooks

- [Face detection notebook](notebooks/train.ipynb) - demonstrates how to train, evaluate and export face detection model.

## Quick Start Guide

In order to see more details please click [here](../QUICK_START_GUIDE.md).
