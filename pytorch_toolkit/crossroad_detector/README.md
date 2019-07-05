# Person Vehicle Bike Detector


## Information

The crossroad detection network model provides detection of 3 class objects: vehicle, pedestrian, non-vehicle (ex: bikes). This detector was trained on the data from crossroad cameras.


## Setup

### Prerequisites

* Ubuntu 16.04
* Python 3.6
* PyTorch 1.0.1
* OpenVINO 2019 R1 with Python API

### Installation

1. Download submodules
```bash
cd openvino_training_extensions
git submodule update --init --recursive
```

2. Create virtual environment
```bash
../object_detection/init_venv.sh
```

3. Activate virtual environment and setup OpenVINO variables
```bash
. ../object_detection/venv/bin/activate
```

7. Build mmdetection module
```bash
../object_detection/prepare_mmdet.sh
```

## Training and evaluation example

**NOTE** To train model on own dataset you should modify `configs/ssd512_mb2_crossroad_clustered.py`.

1. Go to `openvino_training_extensions/pytorch_toolkit/crossroad_detector/` directory

2. The example dataset has annotation in coco and mmdetection CustomDataset format. You can find it here:
   `openvino_training_extensions/pytorch_toolkit/crossroad_detector/dataset`
   To collect CustomDataset annotation used [mmdetection CustomDataset object detection format](https://github.com/open-mmlab/mmdetection/blob/master/GETTING_STARTED.md#use-my-own-datasets). .

3. To start training you have to run:
   ```bash
   ../../external/mmdetection/tools/dist_train.sh configs/ssd512_mb2_crossroad_clustered.py 1    
   ```
   Training artifacts will be stored by default in `ssd512_mb2_crossroad_clustered`

5. Evalution artifacts will be stored by default in `openvino_training_extensions/pytorch_toolkit/crossroad_detector/ssd512_mb2_crossroad_clustered`.
To show results of network model working run
   ```bash
    tensorboard --logdir=./ssd512_mb2_crossroad_clustered
   ``` 

### Demo

```Bash
  python3 ../../external/mmdetection/tools/test.py configs/ssd512_mb2_crossroad_clustered.py ssd512_mb2_crossroad_clustered/epoch_5.pth --show
```

## Conversion to onnx fromat

```bash
python ../object_detection/tools/onnx_export.py configs/ssd512_mb2_crossroad_clustered.py ssd512_mb2_crossroad_clustered/epoch_5.pth ssd512_mb2_crossroad_clustered.onnx
```

## Conversion to Intermediate Representation (IR) of the network

```bash

"${INTEL_OPENVINO_DIR}"/deployment_tools/model_optimizer/mo_onnx.py \
  --model_name ssd512_mb2_crossroad \
  --input_model=ssd512_mb2_crossroad_clustered.onnx \
  --output_dir=./IR \  
```

### Demo

```Bash
python tools/infer_ie.py --model IR/ssd512_mb2_crossroad.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  --label_map dataset/crossroad_label_map.pbtxt \
  dataset/ssd_mbv2_data_val/image_000050.jpg
```
