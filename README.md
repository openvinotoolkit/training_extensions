# Training Toolbox for TensorFlow

Training Toolbox for TensorFlow provides a convenient environment to train
Deep Learning models and convert them using [OpenVINO™
Toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Pre-requisites

- Ubuntu 16.04 / 18.04
- Python 3.4-3.6
- [libturbojpeg](https://github.com/ajkxyz/jpeg4py)
- For Python pre-requisites refer to `requirements.txt`
- (Optional) [TensorFlow GPU pre-requisites](https://www.tensorflow.org/install/gpu).
- (Optional) [OpenVINO™ R3](https://software.intel.com/en-us/openvino-toolkit)
  for exporting of the trained models

# Quick Start Guide

## Setup Training Toolbox for TensorFlow

1. Create virtual environment
<a name="create_venv"></a>

```
cd /<path_to_working_dir>/training_toolbox/tensorflow_toolkit
bash init_venv.sh
```

4. Start to work

  ```
  . venv/bin/activate
  ```

## Tests
In virtual environment run tests:

```
cd /<path_to_working_dir>/training_toolbox/tensorflow_toolkit
nosetests
```

or if you are going to use the OpenVino toolkit:

```
cd /<path_to_working_dir>/training_toolbox/tensorflow_toolkit
export OPEN_VINO_DIR=<PATH_TO_OPENVINO>
nosetests
```

Note: if you have installed the OpenVino toolkit after creating
a virtual environment then you have to [recreate one](#create_venv)
to install required packages for the Model Optimizer into one.

Do not forget to update several environment variables are required to compile and run OpenVINO™ toolkit applications, for details see:
[https://software.intel.com/en-us/articles/OpenVINO-Install-Linux](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux).


## Models
After installation, you are ready to train your own models, evaluate them, use
them for predictions.

* [SSD Object Detection](training_toolbox/ssd_detector)
* [LPRNet](training_toolbox/lpr)
* [Vehicle Attributes](training_toolbox/vehicle_attributes)
