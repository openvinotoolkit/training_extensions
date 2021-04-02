# Vehicle Attributes Recognition: Type and Color

![](./veh_attr.jpg)

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* TensorFlow\* 1.13.1
* OpenVINO™ 2019 R1 with Python API

### Installation

1. Clone and checkout state of `tensorflow/models`:
    ```bash
    git clone https://github.com/tensorflow/models.git $(git rev-parse --show-toplevel)/external/models
    cd $(git rev-parse --show-toplevel)/external/models
    git checkout f0899f18e178afb4b57c50ec32a7e8952e6f6a99
    cd -
    ```

2. Create virtual environment:
    ```bash
    virtualenv venv -p python3.6 --prompt="(veh_attr)"
    ```

3. Activate virtual environment and setup OpenVINO™ variables:
    ```bash
    . venv/bin/activate
    . /opt/intel/openvino/bin/setupvars.sh
    ```
    > **NOTE**: Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
    ```
    echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
    ```

4. Install the modules:
    ```bash
    pip3 install -e .
    pip3 install -e ../utils
    ```

## Train a Vehicle Attributes (VA) Model

To train a Vehicle Attributes model, jump to the
`training_toolbox/vehicle_attributes` directory and find the folder with the sample code
demonstrating how to train a VA model.

We provide predefined configuration for vehicle type and color recognition:
  - Configuration file: [training_toolbox/vehicle_attributes/cars_100/config.py](cars_100/config.py).
  - Trained model: [Vehicle Attributes](https://download.01.org/opencv/openvino_training_extensions/models/vehicle_attributes/vehicle-attributes-barrier-0103.tar.gz).
    A model has two output nodes:
    * type is four components: one for each of 4 types - car, van, truck, bus. The resulting type is `argmax` of these values.
    * color is three components: LAB format normalized from `0` to `1`.

For the demonstration purpose, only a small dataset is uploaded [cars_100](https://download.01.org/opencv/openvino_training_extensions/datasets/vehicle_attributes/cars_100.tar.gz).
It is split into two subsets of images for training and evaluation and serves just as an example of flow.
The resulting model is trained on the internal dataset of more than 50000 images of front-facing vehicles in different
lighting and weather conditions where number of white/black/gray vehicle colors more than twice exceeds others.
It has about 74% of car images, 15% of truck images, 2% of bus images, and 9% of van images.
It was evaluated on the [BitVehicle](http://iitlab.bit.edu.cn/mcislab/vehicledb/) dataset:

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

To train a model, go through the steps described in the sections below.

### Prepare the Dataset

1. Download training data and extract it in the `data/cars_100/images` folder. There are two files with annotations
    named `cars_100_test.json` and `cars_100_train.json` in the `data/cars_100` directory.
    The result structure of the folder should be:
    ```
    ./data/cars_100/
    ├── cars_100_test.json
    ├── cars_100_train.json
    └── images
    ```

2. Make sure that the configuration file `training_toolbox/vehicle_attributes/cars_100/config.py`
    contains correct paths to the annotation files.

3. To use the trained model weights, download it to `training_toolbox/vehicle_attributes` folder, unpack it and set
    the corresponding configuration flag `use_pretrained_weights` to True. Make sure that the correct path
    to the pretrained model is set in the configuration file.

### Training and Evaluation

1.  Open the command line in the `training_toolbox/vehicle_attributes` directory, run the command below:

    ```bash
    python3 tools/train.py cars_100/config.py
    ```

2. In the `training_toolbox/vehicle_attributes` directory, run the command below:
    > **NOTE**: Run in parallel in another terminal, so training and evaluation are performed simultaneously.
    ```bash
    python3 eval.py cars_100/config.py
    ```

3. Training and evaluation artifacts is stored by default in
    `training_toolbox/vehicle_attributes/model`.  To visualize training and evaluation, go to
    `training_toolbox/vehicle_attributes` and run tensorboard with the command below:

    ```bash
    tensorboard --logdir=./model
    ```

    View results in a browser: [http://localhost:6006](http://localhost:6006).

### Export to OpenVINO™

To run the model via OpenVINO™, freeze the TensorFlow graph and
then convert it to OpenVINO™ Intermediate Representation (IR) using the Model Optimizer:

```bash
python3 tools/export.py --data_type FP32 --output_dir <export_path> cars_100/config.py
```

**Default export path**
- `lpr/model/export_<step>/frozen_graph/` - path to the frozen graph
- `lpr/model/export_<step>/IR/<data_type>/` - path to the converted model in the IR format

## Demo

### For the Latest Checkpoint

When the training is complete, model from the checkpoint could be infered on
input data by running `tools/infer.py`:

```
python3 tools/infer_checkpoint.py cars_100/config.py
```

### For the Frozen Graph

```bash
python3 tools/infer.py --model model/export_<step>/frozen_graph/graph.pb.frozen \
    --config cars_100/config.py \
    <image_path>
```

### For the Intermediate Representation (IR)

```bash
python3 tools/infer_ie.py --model model/export_<step>/IR/FP32/vehicle_attributes.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  <image_path>
```
