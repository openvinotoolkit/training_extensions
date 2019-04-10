# Training Toolbox for PyTorch

Training Toolbox for PyTorch provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Pre-requisites

- Ubuntu 16.04 / 18.04
- Python 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python pre-requisites refer to `requirements.txt`
- (Optional) [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  for exporting of the trained models

# Quick Start Guide

## Setup Training Toolbox for PyTorch

1. Create virtual environment
<a name="create_venv"></a>

```
cd /<path_to_working_dir>/training_toolbox/pytorch_toolkit/<model>
bash init_venv.sh
```

2. Start to work

```
. venv/bin/activate
```

Note: if you have installed the OpenVino toolkit after creating
a virtual environment then you have to [recreate one](#create_venv)
to install required packages for the Model Optimizer into one.

Do not forget to update several environment variables are required to compile and run OpenVINO™ toolkit applications, for details see:
[https://software.intel.com/en-us/articles/OpenVINO-Install-Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).


## Models
After installation, you are ready to train your own models, evaluate them, use
them for predictions.

[Face recognition](face_recognition)  
[Human pose estimation](human_pose_estimation)  
[Instance segmentation](instance_segmentation)  
