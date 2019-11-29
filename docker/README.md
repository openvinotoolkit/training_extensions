## Installation

1. [Install docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. [Install NVIDIA plugin for docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
3. [Configure proxy](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)

## Build docker image

* For using `GPU` (base image: nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04)
    ```bash
    bash build.sh gpu
    ```
* For using only `CPU` (base image: ubuntu:18.04)
    ```bash
    bash build.sh cpu
    ```

> **NOTE** some models supports training only on GPU

## Run docker image

* For using `GPU`
    ```bash
    bash run.sh gpu
    ```
* For using only `CPU`
    ```bash
    bash run.sh cpu
    ```

> **NOTE** repository directory will be mounted to docker container as `/workspace`, in result all artifacts from training process will be available on host machine

To train some model follow instructions from `README.md`

## Example

Training [License Plate Recognition](../tensorflow_toolkit/lpr)

```bash
bash build.sh gpu
bash run.sh gpu

cd $(git rev-parse --show-toplevel)/tensorflow_toolkit/lpr

virtualenv venv -p python3 --prompt="(lpr)"
echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate

pip3 install -e .
pip3 install -e ../utils

bash ../prepare_modules.sh

cd $(git rev-parse --show-toplevel)/data/synthetic_chinese_license_plates

wget https://download.01.org/opencv/openvino_training_extensions/datasets/license_plate_recognition/Synthetic_Chinese_License_Plates.tar.gz
tar xf Synthetic_Chinese_License_Plates.tar.gz

python3 make_train_val_split.py Synthetic_Chinese_License_Plates/annotation

cd $(git rev-parse --show-toplevel)/tensorflow_toolkit/lpr

python3 tools/train.py chinese_lp/config.py

python3 tools/export.py --data_type FP32 --output_dir model/export chinese_lp/config.py
```

IR of model will be available by `$(git rev-parse --show-toplevel)/tensorflow_toolkit/lpr/model/export/IR/FP32/` on host machine
