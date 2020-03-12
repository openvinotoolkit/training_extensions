# :warning: **Deprecated** :warning:

Recommend use [pytorch_toolkit/object_detection](../../pytorch_toolkit/object_detection)

---

# SSD Object Detection

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.6
* TensorFlow\* 1.10.0
* OpenVINO™ 2019 R1 with Python API

### Installation

1. Create virtual environment:
    ```bash
    virtualenv venv -p python3 --prompt="(ssd)"
    ```

2. Activate virtual environment and setup OpenVINO™ variables:
    ```bash
    . venv/bin/activate
    . /opt/intel/openvino/bin/setupvars.sh
    ```
    > **TIP** Good practice is adding `. /opt/intel/openvino/bin/setupvars.sh` to the end of the `venv/bin/activate`.
    ```
    echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
    ```

3. Install the modules:
    ```bash
    pip3 install -e .
    pip3 install -e ../utils
    ```
    For the case without GPU, use the `CPU_ONLY=true` environment variable:
    ```bash
    CPU_ONLY=true pip3 install -e .
    pip3 install -e ../utils
    ```

3. Download and prepare required submodules:
    ```bash
    bash ../prepare_modules.sh
    ```

## Train an SSD Detection Model

To train a [Single Shot Detector](https://arxiv.org/abs/1512.02325), go to the
`tensorflow_toolkit/ssd_detector` directory. The `ssd_detector` folder with sample code
demonstrates how to train a MobileNetV2-based SSD object detector.

We provide 2 predefined configurations:
* Vehicles and license plates detector
  ![VLP detection](vlp/docs/sample.jpg "Example of VLP detector inference")

  - Configuration file: [vlp/config.py](vlp/config.py)
  - Trained model: [MobileNet v2 0.35 256x256](https://download.01.org/opencv/openvino_training_extensions/models/ssd_detector/ssd-mobilenet-v2-0.35.1-barrier-256x256-0123.tar.gz)

* Object detector trained on the [COCO dataset](../../data/coco/README.md)
  - Configuration file: [tensorflow_toolkit/ssd_detector/coco/config.py](coco/config.py)
  - Trained model: [MobileNet v2 1.0 256x256](https://www.myqnapcloud.com/smartshare/6d62i0464l6p7019t3wz2891_6ku3ACR)

### Quick Start with Vehicles and License Plates Detector

### Prepare a Dataset

The sample model learns to detect vehicles and license plates on the
[BitVehicle](http://iitlab.bit.edu.cn/mcislab/vehicledb/) dataset.

To train a model, go through the following steps:

1. Download training data and put it in the `data/bitvehicle` directory
    according to the [data/bitvehicle/README.md](../../data/bitvehicle/README.md)
    file. Annotation files in the **COCO** format (refer to
    [cocodataset](http://cocodataset.org/#format-data) for details) are already
    located in `data/bitvehicle`.

    The resulting structure of the folder:
    ```
    ./data/bitvehicle/
    ├── images/
    │   ├── vehicle_0000001.jpg
    |   ...
    ├── bitvehicle_test.json
    ├── bitvehicle_train.json
    └── README.md
    ```

2. Change `annotation_path` in `vlp/config.py` to `data/bitvehicle` instead
    of test dataset `data/vlp_test/`.

3. If necessary, modify training settings by editing
    [vlp/config.py](vlp/config.py) or leave them by
    default. For details, read comments in
    [config.py](vlp/config.py). Notable parameters in the `train`
    class are:
     * `batch_size` - number of images in training batch. The default value is `32`, but can be increased or decreased depending on the amount of
       available memory.
     * `annotation_path` - path to a JSON file with an annotation in the COCO format
       The default value is the relative path to the bitvehicle annotation, but you
       can use your own annotation.
     * `steps` - number of training iterations
     * `execution.CUDA_VISIBLE_DEVICES` - Environment variable to control the CUDA
       device used for training. The default value is `0`. In case you have
       multiple GPUs, change it to the respective number, or leave this
       string empty if you want to train on CPU.
     * `cache_type` - type of input data to save in cache to speed up data
       loading and preprocessing. The default value is `NONE`.
       > **NOTE**: Caching might cause system slowdown, so if you do not have
       enough RAM memory, disable cashing by passing `NONE` to this parameter.

### Train and Evaluate

1. To start training, go to the `tensorflow_toolkit/ssd_detector` directory and run the command below:

    ```
    python3 tools/train.py vlp/config.py
    ```

    > **TIP**: To start from a pretrained checkpoint, use `initial_weights_path` in `config.py`.

2. To start the evaluation process, go to the `tensorflow_toolkit/ssd_detector` directory and run the command below:

    ```
    python3 tools/eval.py vlp/config.py
    ```

    > **NOTE**: Take the step 4 in another terminal, so training and
    evaluation are performed simultaneously.

3. Training and evaluation artifacts are stored by default in
    `tensorflow_toolkit/ssd_detector/vlp/model`. To visualize training and evaluation, go to
    `tensorflow_toolkit/ssd_detector/vlp` and run tensorboard with the command below:

    ```
    tensorboard --logdir=./model
    ```

    Then view results in a browser: [http://localhost:6006](http://localhost:6006).

    ![BitVehicle TensorBoard](vlp/docs/tensorboard.png "TensorBoard for BitVehicle training")

### Export to OpenVINO™

To run the model via OpenVINO™, freeze the TensorFlow graph and convert it to the OpenVINO™ Internal Representation (IR) using the Model Optimizer:

```
python3 tools/export.py --data_type FP32 --output_dir vlp/model/export vlp/config.py
```

The script results in three new artifacts:

**Default export path**
- `vlp/model/export_<step>/frozen_graph/` - path to the frozen graph
- `vlp/model/export_<step>/IR/<data_type>/` - path to the converted model in the IR format

## Demo

### For the Latest Checkpoint

When the training is complete, you can infer the model from the checkpoint on
input data by running `tensorflow_toolkit/ssd_detector/infer.py`:

```
python3 infer_checkpoint.py vlp/config.py --video --input=<path_to_input_video> --show
python3 infer_checkpoint.py vlp/config.py --json --input=<path_to_annotation_json> --show
```

### For a Frozen Graph

```Bash
python3 tools/infer.py --model vlp/model/export/frozen_graph/graph.pb.frozen \
    <image_path>
```

### For Intermediate Representation (IR)

```Bash
python3 tools/infer_ie.py --model vlp/model/export/frozen_graph/graph.pb.frozen \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  --input_type json \
  --input ../../data/vlp_test/annotations_test.json \
  --dump_predictions_to_json True \
  --output_json_path test.json
```

A model in the IR format could be inferred using the Python sample from OpenVINO™ located at `<path_to_computer_vision_sdk>/inference_engine/samples/python_samples/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py`

```bash
python3 object_detection_demo_ssd_async.py \
    -m <path_to_converted_model>/graph.xml \
    -l <path_to_computer_vision_sdk>/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so \
    -i <path_to_input_video>
```
