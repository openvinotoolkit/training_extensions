# Person Bike Vehicle Detector


## Information

The crossroad detection network model provides detection of 3 class objects: vehicle, pedestrian, non-vehicle (ex: bikes). This detector was trained on the data from crossroad cameras.


## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.6
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
virtualenv venv -p python3 --prompt="(pbv)"
```

3. Modify `venv/bin/activate` to set environment variables
```
cat <<EOT >> venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:$(git rev-parse --show-toplevel)/external/models/research
export PYTHONPATH=\$PYTHONPATH:$(git rev-parse --show-toplevel)/external/models/research/slim
. /opt/intel/openvino/bin/setupvars.sh
EOT
```

4. Activate virtual environment and setup OpenVINO variables
```bash
. venv/bin/activate
```

5. Install modules
```
pip3 install -r requirements.txt
pip3 install -e ../utils
```

6. Build and install COCO API for python
```bash
cd $(git rev-parse --show-toplevel)/external/cocoapi
2to3 . -w
cd PythonAPI
make install
```

## Training and evaluation example

**NOTE** To train model on own dataset you should change `num_steps: 10` in `configs/pipeline.config`.

1. Go to `openvino_training_extensions/tensorflow_toolkit/veh_ped_nonveh_ssd_mobilenetv2_detector/` directory

2. The example dataset has annotation in coco format. You can find it here:
   `openvino_training_extensions/tensorflow_toolkit/veh_ped_nonveh_ssd_mobilenetv2_detector/dataset`
   To collect annotation used [COCO object detection format](http://cocodataset.org/#format-data). .

3. To convert the dataset to tfrecords you have to run:
   ```bash
   python ./tools/create_crossroad_extra_tf_records.py \
       --train_image_dir=./dataset/ssd_mbv2_data_train \
       --val_image_dir=./dataset/ssd_mbv2_data_val/ \
       --train_annotations_file=./dataset/annotation_example_train.json \
       --val_annotations_file=./dataset/annotation_example_val.json \
       --output_dir=dataset/tfrecords
   ```

4. To start training you have to run:
   ```bash
   python ../../external/models/research/object_detection/model_main.py \
     --model_dir=./model \
     --pipeline_config_path=./configs/pipeline.config
   ```
   Training artifacts will be stored by default in `model`

5. Evalution artifacts will be stored by default in `openvino_training_extensions/tensorflow_toolkit/veh_ped_nonveh_ssd_mobilenetv2_detector/model/eval_0/`.
To show results of network model working run
   ```bash
   tensorboard --logdir=./model
   ```

   And view results in a browser: [http://localhost:6006](http://localhost:6006).
   ![](./tb_eval.png)

## Freezing graph

```Bash
python ../../external/models/research/object_detection/export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path=./configs/pipeline.config \
  --trained_checkpoint_prefix=./model/model.ckpt-10 \
  --output_directory ./model/export_10
```

### Demo

```Bash
python tools/infer.py --model=model/export_10/frozen_inference_graph.pb \
  --label_map=dataset/crossroad_label_map.pbtxt \
  dataset/ssd_mbv2_data_val/image_000000.jpg
```

## Conversion to Intermediate Representation (IR) of the network

```bash
"${INTEL_OPENVINO_DIR}"/deployment_tools/model_optimizer/mo_tf.py \
  --model_name veh_ped_nonveh_ssd_mobilenetv2_detector \
  --input_model=./model/export_10/frozen_inference_graph.pb \
  --output_dir=./model/export_10/IR \
  --tensorflow_object_detection_api_pipeline_config=./configs/pipeline.config \
  --tensorflow_use_custom_operations_config="${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json"
```

### Demo

```Bash
python tools/infer_ie.py --model model/export_10/IR/veh_ped_nonveh_ssd_mobilenetv2_detector.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  --label_map dataset/crossroad_label_map.pbtxt \
  dataset/ssd_mbv2_data_val/image_000000.jpg
```
