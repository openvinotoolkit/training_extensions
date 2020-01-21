# Person Vehicle Bike Detector

The crossroad-detection network model provides detection of three class objects: vehicle, pedestrian, non-vehicle (like bikes). This detector was trained on the data from crossroad cameras.


## Training and Evaluation Example

> **NOTE**: To train model on your own dataset, modify `configs/person_vehicle_bike_sd512_mb2_clustered.py`.

1. Go to the `openvino_training_extensions/pytorch_toolkit/object_detection/` directory.

2. The example dataset has annotation in the Common Objects in Context (COCO) and mmdetection CustomDataset format. You can find it here in 
   `openvino_training_extensions/data/airport`.
   To collect CustomDataset annotation, use [mmdetection CustomDataset object detection format](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md#use-my-own-datasets). 

3. Download pretrained [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/person_vehicle_bike_sd512_mb2_clustered_epoch_21.pth) to `object_detection/checkpoint`.

4. To start training, run the following:
    ```bash
    ../../external/mmdetection/tools/dist_train.sh configs/person_vehicle_bike_sd512_mb2_clustered.py 1    
    ```
   Training artifacts will be stored by default in `person_vehicle_bike_sd512_mb2_clustered`

5. Evaluation artifacts are stored by default in `models/person_vehicle_bike_sd512_mb2_clustered`.
   To see results of network model working, run the following:
   ```bash
   tensorboard --logdir=./models/person_vehicle_bike_sd512_mb2_clustered
   ```

### Demo

```Bash
python ../../external/mmdetection/tools/test.py \
  configs/person_vehicle_bike_sd512_mb2_clustered.py \
  models/person_vehicle_bike_sd512_mb2_clustered/epoch_5.pth \
  --show
```

## Conversion to ONNX\* Format

```bash
python tools/onnx_export.py \
  configs/person_vehicle_bike_sd512_mb2_clustered.py \
  models/person_vehicle_bike_sd512_mb2_clustered/epoch_5.pth \
  person_vehicle_bike_sd512_mb2_clustered.onnx
```

## Conversion to Intermediate Representation (IR) of the Network

```bash
mo.py --input_model=person_vehicle_bike_sd512_mb2_clustered.onnx \
  --scale 255 \
  --reverse_input_channels \
  --output_dir=./IR \
  --data_type=FP32
```

### Demo

```Bash
python tools/infer_ie.py --model IR/person_vehicle_bike_sd512_mb2_clustered.xml \
  --device=CPU \
  --cpu_extension="${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so" \
  ../../data/airport/val/image_000009.jpg
```
