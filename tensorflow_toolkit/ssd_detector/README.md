# SSD Object Detection

## Setup


### Prerequisites

* Ubuntu 16.04
* Python 3.6
* TensorFlow 1.10.0
* OpenVINO 2019 R1 with Python API


### Installation

1. Create virtual environment
```bash
virtualenv venv -p python3 --prompt="(ssd)"
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
For case without GPU use `CPU_ONLY=true` environment variable
```bash
CPU_ONLY=true pip3 install -e .
pip3 install -e ../utils
```

3. Download and prepare required submodules
```bash
bash ../prepare_modules.sh
```

## Train SSD detection model

To train a [Single Shot Detector](https://arxiv.org/abs/1512.02325), jump to
training_toolbox/ssd_detector directory. You'll see the `ssd_detector` folder with sample code
demonstrating how to train a MobileNetV2-based SSD object detector.

We provide 2 predefined configurations:
* Vehicles and license plates detector.
  ![VLP detection](vlp/docs/sample.jpg "Example of VLP detector inference")

  - Configuration file: [training_toolbox/ssd_detector/vlp/config.py](vlp/config.py).
  - Trained model: [MobileNet v2 0.35 256x256](https://download.01.org/opencv/openvino_training_extensions/models/ssd_detector/ssd-mobilenet-v2-0.35.1-barrier-256x256-0123.tar.gz).

* Object detector trained on the [COCO dataset](../../data/coco/README.md).
  - Configuration file: [training_toolbox/ssd_detector/coco/config.py](coco/config.py).
  - Trained model: [MobileNet v2 1.0 256x256](https://www.myqnapcloud.com/smartshare/6d62i0464l6p7019t3wz2891_6ku3ACR).

### Quck start with vehicles and license plates detector

### Prepare dataset

The sample model will learn how to detect vehicles and license plates on
[BitVehicle](http://iitlab.bit.edu.cn/mcislab/vehicledb/) dataset.

To train a model, go through the following steps:

1. Download training data and put it in the `data/bitvehicle` directory
    according to [data/bitvehicle/README.md](../../data/bitvehicle/README.md)
    file. Annotation files in the **COCO** format (refer to
    [cocodataset](http://cocodataset.org/#format-data) for details) are already
    located in `data/bitvehicle`.

    The result structure of the folder should be:
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
    of test dataset `data/vlp_test/`

3. If necessary, you can modify training settings by editing
    [training_toolbox/ssd_detector/vlp/config.py](vlp/config.py) or leave them by
    default. For more details please read comments in
    [config.py](vlp/config.py). Notable parameters in `train`
    class are:
     * `batch_size` - number of images in training batch, by default it's set to
       `32`, but could be increased or decreased depending on the amount of
       available memory.
     * `annotation_path` - path to json file with annotation in the **COCO** format,
       by default it's set to relative path to bitvehicle annotation, but you
       could use your own annotation.
     * `steps` - number of training iterations
     * `execution.CUDA_VISIBLE_DEVICES` - Environment variable to control cuda
       device used for training. By default, it's set to `0`. In case you have
       multiple GPUs, you can change it to the respective number, or leave this
       string empty, if you want to train on CPU.
     * `cache_type` - type of input data to save in cache to speed-up data
       loading and preprocessing. By default it's set to `ENCODED`.
       Remember that caching might cause system slowdown, so if you don't have
       enough RAM memory better to disable it, pass `NONE` to this parameter.


### Train and evaluation

1. To start training go to `training_toolbox/ssd_detector` directory and type in command line:

    ```
    python3 tools/train.py vlp/config.py
    ```

    **NOTE** To start from pretrained checkpoint use `initial_weights_path` in `config.py`

2. To start evaluation process go to `training_toolbox/ssd_detector` directory and type
    in command line:

    ```
    python3 tools/eval.py vlp/config.py
    ```

    Do step 4 in another terminal, so training and evaluation are performed simultaneously.

3. Training and evaluation artifacts will be stored by default in
    `training_toolbox/ssd_detector/vlp/model`.  To visualize training and evaluation, go to
    `training_toolbox/ssd_detector/vlp` and run tensorboard with:

    ```
    tensorboard --logdir=./model
    ```

    And view results in a browser: [http://localhost:6006](http://localhost:6006).

    ![BitVehicle TensorBoard](vlp/docs/tensorboard.png "TensorBoard for BitVehicle training")


### Export to OpenVINO

To run the model via OpenVINO one has to freeze TensorFlow graph and
then convert it to OpenVINO Internal Representation (IR) using Model Optimizer:

```
python3 tools/export.py --data_type FP32 --output_dir vlp/model/export vlp/config.py
```

As a result, you'll find three new artifacts:  
**default export path**
- `vlp/model/export_<step>/frozen_graph/` - path to frozen graph
- `vlp/model/export_<step>/IR/<data_type>/` - path to converted model in IR format


## Demo

### For the latest checkpoint

When training is complete, model from the checkpoint could be infered on
input data by running `training_toolbox/ssd_detector/infer.py`:

```
python3 infer_checkpoint.py vlp/config.py --video --input=<path_to_input_video> --show
python3 infer_checkpoint.py vlp/config.py --json --input=<path_to_annotation_json> --show
```

### For frozen graph
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

Model in IR format could be infered using python sample from OpenVINO™ which
could be found here: `<path_to_computer_vision_sdk>/inference_engine/samples/python_samples/object_detection_demo_ssd_async/object_detection_demo_ssd_async.py`

```
python3 object_detection_demo_ssd_async.py -m <path_to_converted_model>/graph.xml -l <path_to_computer_vision_sdk>/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so -i <path_to_input_video>
```
