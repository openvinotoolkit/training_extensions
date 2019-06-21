# LPRNet: License Plate Recognition

![](./lpr.png)

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.6
* TensorFlow 1.13.1
* OpenVINO 2019 R1 with Python API


### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(lpr)"
```

2. Activate virtual environment and setup OpenVINO variables
```bash
. venv/bin/activate
. /opt/intel/openvino/bin/setupvars.sh
```
**NOTE** Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
```
echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
```

3. Install the modules
```bash
pip3 install -e .
pip3 install -e ../utils
```


## Train LPRNet model

To train a [LPRNet](https://arxiv.org/abs/1806.10447), jump to
training_toolbox/lpr directory. You'll see the folder with sample code
demonstrating how to train a LPRNet model.

We provide predefined configuration for:
* Chinese license plates recognition.
  - Configuration file: [training_toolbox/lpr/chinese_lp/config.py](chinese_lp/config.py).
  - Trained model: [LPRNet 94x24](https://download.01.org/opencv/openvino_training_extensions/models/license_plate_recognition/license-plate-recognition-barrier-0007.tar.gz).

As training dataset for this model [Synthetic Chinese License Plates](https://download.01.org/opencv/openvino_training_extensions/datasets/license_plate_recognition/Synthetic_Chinese_License_Plates.tar.gz) dataset was used.

To train a model, go through the following steps:


### Prepare dataset

1. Download training data and extract it in `data/synthetic_chinese_license_plates` folder. After extracting it will
    consist from folder with training images named `crops` and text file with annotations named `annotation`.

2. After extracting training data archive run python script from
    `data/synthetic_chinese_license_plates/make_train_val_split.py` to make split of
    the whole annotations into `train` and `val` feeding him path to `data/synthetic_chinese_license_plates/annotation`
    file from archive as an input. As a result you'll find `data/synthetic_chinese_license_plates/train`,
    `data/synthetic_chinese_license_plates/val` annotation files with full path to images and labels in the folder
    with extracted data.

    ```bash
    python3 make_train_val_split.py data/synthetic_chinese_license_plates/annotation
    ```

    The result structure of the folder should be:
    ```
    ./data/synthetic_chinese_license_plates/
    ├── make_train_val_split.py
    └── Synthetic_Chinese_License_Plates/
        ├── annotation
        ├── crops/
        │   ├── 000000.png
        |   ...
        ├── LICENSE
        ├── README
        ├── train
        └── val
    ```

3. Then edit `training_toolbox/lpr/chinese_lp/config.py` by pointing out
    `train.file_list_path` and `eval.file_list_path`
    parameters in train section to paths of obtained `train` and `val`
    annotation files accordingly.


### Train and evaluation

1. To start training process type in command line:
    ```bash
    python3 tools/train.py chinese_lp/config.py
    ```

    To start from pretrained checkpoint type in command line:
    ```bash
    python3 tools/train.py chinese_lp/config.py \
      --init_checkpoint <data_path>/license-plate-recognition-barrier-0007/model.ckpt
    ```

2. To start evaluation process type in command line:
    ```bash
    python3 tools/eval.py chinese_lp/config.py
    ```

    **NOTE** Before doing step 4, make sure that parameter `eval.file_list_path` in
    `lpr/chinese_lp/config.py` pointing out to file with
    annotations to test on. Do step 4 in another terminal, so training and
    evaluation are performed simultaneously.


3. Training and evaluation artifacts will be stored by default in `lpr/chinese_lp/model`.
   To visualize training and evaluation, run vino`tensorboard` with:

    ```bash
    tensorboard --logdir=./model
    ```

    And view results in a browser: [http://localhost:6006](http://localhost:6006).


### Export to OpenVINO

To run the model via OpenVINO one has to freeze TensorFlow graph and
then convert it to OpenVINO Internal Representation (IR) using Model Optimizer:

```Bash
python3 tools/export.py --data_type FP32 --output_dir <export_path> chinese_lp/config.py
```

**default export path**:  
`lpr/model/export_<step>/frozen_graph` - path to frozen graph
`lpr/model/export_<step>/IR/<data_type>` - path to converted model in IR format

## Demo

### For the latest checkpoint

**NOTE** Input data for infer should be set via parameter `infer.file_list_path` in
`training_toolbox/lpr/chinese_lp/config.py` and must be look like text file
with list of path to license plates images in format:
```
path_to_lp_image1
path_to_lp_image2
...
```

When training is complete, model from the checkpoint could be infered on
input data by running `training_toolbox/lpr/chinese_lp/infer.py`:

```Bash
python3 tools/infer_checkpoint.py chinese_lp/config.py
```

### For frozen graph
```Bash
python3 tools/infer.py --model model/export_<step>/frozen_graph/graph.pb.frozen \
    --config chinese_lp/config.py \
    <image_path>
```

### For Intermediate Representation (IR)
```Bash
python3 tools/infer_ie.py --model model/export_<step>/IR/FP32/lpr.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  --config chinese_lp/config.py \
  <image_path>
```


## Citation

If you find *LPRNet* useful in your research, please, consider to cite the following paper:

```
@article{icv2018lprnet,
title={LPRNet: License Plate Recognition via Deep Neural Networks},
author={Sergey Zherzdev and Alexey Gruzdev},
journal={arXiv:1806.10447},
year={2018}
}
```
