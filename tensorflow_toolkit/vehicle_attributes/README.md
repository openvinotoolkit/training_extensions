# Vehicle attributes recognition: type and color

![](./veh_attr.jpg)

## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.6
* TensorFlow 1.13.1
* OpenVINO 2019 R1 with Python API


### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(veh_attr)"
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


## Train Vehicle Attributes model

To train a Vehicle Attributes model, jump to
training_toolbox/vehicle_attributes directory. You'll see the folder with a sample code
demonstrating how to train a VA model.

We provide predefined configuration for:
* Vehicle type and color recognition.
  - Configuration file: [training_toolbox/vehicle_attributes/cars_100/config.py](cars_100/config.py).
  - Trained model: [Vehicle Attributes](https://download.01.org/opencv/openvino_training_extensions/models/vehicle_attributes/vehicle-attributes-barrier-0103.tar.gz).
    A model has two output nodes:
    . type is 4 components: one for each of 4 types - car, van, truck, bus. The resulting type is argmax of these values.
    . color is 3 components: LAB format normalized from 0 to 1

For demonstration purpose only a small dataset is uploaded [cars_100](https://download.01.org/opencv/openvino_training_extensions/datasets/vehicle_attributes/cars_100.tar.gz).
It is divided by two subsets of images for training and evaluation and serves just as an example of flow.
The resulting model is trained on the internal dataset of more than 50000 images of front-facing vehicles in different
lighting and weather conditions where number of white/black/gray vehicle colors more than twice exceeds others.
It has about 74% car images, 15% truck images, 2% bus images and 9% van images.
It was evaluated on [BitVehicle](http://iitlab.bit.edu.cn/mcislab/vehicledb/) dataset:

| Metric                                    | Value    |
|-------------------------------------------|----------|
| Color mean absolute error                 | 9.076    |
| Mean absolute error of l color component  | 19.75    |
| Mean absolute error of a color component  | 3.353    |
| Mean absolute error of b color component  | 4.129    |
| Type accuracy - average                   | 0.958    |
| Type accuracy - car                       | 0.946    |
| Type accuracy - van                       | 0.967    |
| Type accuracy - truck                     | 0.952    |
| Type accuracy - bus                       | 0.966    |

To train a model, go through the following steps:

### Prepare dataset

1. Download training data and extract it in `data/cars_100/images` folder. There are two files with annotations
    named `cars_100_test.json` and `cars_100_train.json` in the 'data/cars_100' directory.
    The result structure of the folder should be:
    ```
    ./data/cars_100/
    ├── cars_100_test.json
    ├── cars_100_train.json
    └── images
    ```

2. Please make sure that configuration fila `training_toolbox/vehicle_attributes/cars_100/config.py`
    contains correct paths to the annotation files.

3. To use the trained model weights, download it to `training_toolbox/vehicle_attributes` folder, unpack it and set
    the corresponding configuration flag 'use_pretrained_weights' to True. Please make sure that the correct path
    to pretrained model is set in the configuration file.


### Train and evaluation

1. To start training go to `training_toolbox/vehicle_attributes` directory and type in command line:

    ```bash
    python3 tools/train.py cars_100/config.py
    ```

2. To start evaluation process go to `training_toolbox/vehicle_attributes` directory and type
    in command line:
    **NOTE** Run in in parallel in another terminal, so training and evaluation are performed simultaneously.
    ```bash
    python3 eval.py cars_100/config.py
    ```

3. Training and evaluation artifacts will be stored by default in
    `training_toolbox/vehicle_attributes/model`.  To visualize training and evaluation, go to
    `training_toolbox/vehicle_attributes` and run tensorboard with:

    ```bash
    tensorboard --logdir=./model
    ```

    And view results in a browser: [http://localhost:6006](http://localhost:6006).


### Export to OpenVINO
To run the model via OpenVINO one has to freeze TensorFlow graph and
then convert it to OpenVINO Internal Representation (IR) using Model Optimizer:

```Bash
python3 tools/export.py --data_type FP32 \
  --mo_config chinese_lp/mo.yaml \
  chinese_lp/config.py
```

As a result, you'll find three new artifacts:
`lpr/model/export_<step>/frozen_graph` - path to frozen graph
`lpr/model/export_<step>/IR/<data_type>` - path to converted model in IR format

## Demo

### For the latest checkpoint

When training is complete, model from the checkpoint could be infered on
input data by running `tools/infer.py`:

```
python3 tools/infer.py cars_100/config.py
```

### For frozen graph
```Bash
python3 tools/infer.py --model model/export_<step>/frozen_graph/graph.pb.frozen \
    --config cars_100/config.py \
    <image_path>
```

### For Intermediate Representation (IR)
```Bash
python3 tools/infer_ie.py --model model/export_<step>/IR/FP32/vehicle_attributes.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  <image_path>
```
