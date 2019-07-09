# SSD MobileNet FPN 602

This training extension is designed for running SSD MobileNet with FPN on Intel devices which can not place the original 640-size model. By resizing and fine-tuning the network, the model is consistent with OpenVINO running on various Intel devices as well as achieves decent accuracy close to the original 640-size model.


## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.5
* TensorFlow 1.13.1
* OpenVINO 2019 R1 with Python API

### Installation

1. Download submodules
```bash
cd openvino_training_extensions
git submodule update --init --recursive
```

2. Create virtual environment
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
virtualenv venv -p python3 --prompt="(ssd_mobilenet_fpn_602)"
```

3. Modify `venv/bin/activate` to set environment variables
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
cat <<EOT >> venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:$(git rev-parse --show-toplevel)/external/models/research
export PYTHONPATH=\$PYTHONPATH:$(git rev-parse --show-toplevel)/external/models/research/slim
. /opt/intel/openvino/bin/setupvars.sh
EOT
```

4. Activate virtual environment and setup OpenVINO variables
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
. venv/bin/activate
```

5. Install modules
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
pip3 install -r requirements.txt
```

6. Build and install COCO API for python
```bash
cd $(git rev-parse --show-toplevel)/external/cocoapi
2to3 . -w
cd PythonAPI
make
cp -r pycocotools ../../models/research
```

7. Protobuf Compilation
```bash
# From openvino_training_extensions/external/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```

## Data Preparation
1. Please download images and annotations from [cocodataset.org](cocodataset.org/#download). COCO2017 is used in this repo for training and validation.
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
mkdir -p dataset/images
wget -P dataset http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -P dataset/images http://images.cocodataset.org/zips/train2017.zip
wget -P dataset/images http://images.cocodataset.org/zips/val2017.zip
unzip dataset/annotations_trainval2017.zip -d dataset
unzip dataset/train2017.zip -d dataset/images
unzip dataset/val2017.zip -d dataset/images
```

2. Here a data generation script is provided to generate training and validation set since [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is using COCO minival set(note that the split is different from COCO2017 val) for evaluation. See [create_coco_tfrecord.py](tools/create_coco_tfrecord.py) for more details. To run the script,
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
python tools/create_coco_tfrecord.py \
--image_folder=dataset/images \
--annotation_folder=dataset/annotations \
--coco_minival_ids_file=../../external/models/research/object_detection/data/mscoco_minival_ids.txt \
--output_folder=dataset/tfrecord
```

3. After data preparation, tfrecords are located at **dataset/tfrecord**. Files with *coco_train2017_plus.record* and *coco_minival2017.record* prefixes can be used for training and validatation respectively.


## Training and Evaluation

1. Download pre-trained model from TensorFlow Object Detection Model Zoo and extract it.
```bash
# From openvino_training_extensions/tensorflow_toolkit/ssd_mobilenet_fpn_602/
mkdir -p models
wget -P models http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar xzvf models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz -C models
```

2. Configure settings in *configs/pipeline.config*. Replace `PATH_TO_BE_CONFIGURED` with corresponding path.

3. Run training as follows.
```bash
python ../../external/models/research/object_detection/model_main.py \
--model_dir=./models/checkpoint \
--pipeline_config_path=./configs/pipeline.config
```

4. Run following command visualization, and follow the terminal instruction to view training and evaluation result in browser.
```bash
tensorboard --logdir=./model
```

## Model Conversion

1. Convert TF checkpoints into frozen inference graph.
2. Convert TF frozen inference graph in OpenVINO IR.

## OpenVINO Demo
