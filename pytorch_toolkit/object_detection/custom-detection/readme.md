# Custom object detector

Custom object detectors are lightweight object detection models that have been pre-trained on MS COCO object detection dataset.
It is assumed that one will use these pre-trained models as starting points in order to train specific object detection models (e.g. 'cat' and 'dog' detection).
There was no a goal to train lightweight ready-to-use 80 class (MS COCO classes) detector.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- |
| mobilenet_v2-2s_ssd-256x256 | 0.86 | 1.99 | 11.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-256x256.pth), [configuration file](./mobilenet_v2-2s_ssd-256x256/config.py) | 3 |
| mobilenet_v2-2s_ssd-384x384 | 1.92 | 1.99 | 13.3 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-384x384.pth), [configuration file](./mobilenet_v2-2s_ssd-384x384/config.py) | 3 |
| mobilenet_v2-2s_ssd-512x512 | 3.42 | 1.99 | 12.7 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-512x512.pth), [configuration file](./mobilenet_v2-2s_ssd-512x512/config.py) | 3 |

## Training pipeline

### 0. Change a directory in your terminal to object_detection.

```bash
cd <openvino_training_extensions>/pytorch_toolkit/object_detection
```

### 1. Select a training configuration file and get pre-trained snapshot if available. Please see the table above.

```bash
export MODEL_NAME=mobilenet_v2-2s_ssd-256x256
export CONFIGURATION_FILE=./custom-detection/$MODEL_NAME/config.py
```

### 2. Collect dataset

You can train a model on existing toy dataset `openvino_training_extensions/data/airport`. Obviously such dataset is not sufficient for training good enough model.

### 3. Prepare annotation

The existing toy dataset has annotation in the Common Objects in Context (COCO) and mmdetection CustomDataset format.

### 4. Training

Since there are model templates rather than ready-to-use models (though technically one can use the as they are) it is needed to update existing configuration file.
It can be done by `--update_args` parameter or modifications inside configuration file.
```bash
export NUM_CLASSES=3
export CLASSES="vehicle,person,non-vehicle"
export WORK_DIR="my_custom_detector"
export UPDATE_CONFIG="model.bbox_head.num_classes=${NUM_CLASSES} \
                      data.train.dataset.classes=${CLASSES} \
                      data.val.classes=${CLASSES} \
                      data.val.classes=${CLASSES} \
                      total_epochs=20 \
                      resume_from=${MODEL_NAME}.pth \
                      work_dir=${WORK_DIR}"
```

* To train the detector on a single GPU, run in your terminal:

   ```bash
   python ../../external/mmdetection/tools/train.py \
            $CONFIGURATION_FILE \
            --update_config $UPDATE_CONFIG
   ```

* To train the detector on multiple GPUs, run in your terminal:

   ```bash
   ../../external/mmdetection/tools/dist_train.sh \
            $CONFIGURATION_FILE \
            <GPU_NUM> \
            --update_config $UPDATE_CONFIG
   ```

### 5. Validation

To dump detection of your model as well as compute MS-COCO metrics run:

```bash
python ../../external/mmdetection/tools/test.py \
        ${WORK_DIR}/config.py \
        <CHECKPOINT> \
        --out result.pkl \
        --eval bbox \
        --update_config $UPDATE_CONFIG
```

### 6. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python ../../external/mmdetection/tools/export.py \
      ${WORK_DIR}/config.py \
      <CHECKPOINT> \
      <EXPORT_FOLDER> \
      openvino
```

This produces model `config.xml` and weights `config.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

For SSD networks an alternative OpenVINO™ representation is possible.
To opt for it use extra `--alt_ssd_export` key to the `export.py` script.
SSD model exported in such way will produce a bit different results (non-significant in most cases),
but it also might be faster than the default one. As a rule SSD models in [Open Model Zoo](https://github.com/opencv/open_model_zoo/) are exported using this option.

### 7. Validation of IR

Instead of running `test.py` you need to run `test_exported.py` and then repeat steps listed in [Validation paragraph](#5-validation).

```bash
python ../../external/mmdetection/tools/test_exported.py  \
      ${WORK_DIR}/config.py \
      <EXPORT_FOLDER>/config.xml \
      --out results.pkl \
      --eval bbox
```

### 8. Demo

To see how the converted model works using OpenVINO you need to run `test_exported.py` with `--show` option.

```bash
python ../../external/mmdetection/tools/test_exported.py  \
      ${WORK_DIR}/config.py \
      <EXPORT_FOLDER>/config.xml \
      --show
```

## Other

### Theoretical computational complexity estimation

To get per-layer computational complexity estimations, run the following command:

```bash
python ../../external/mmdetection/tools/get_flops.py \
      ${WORK_DIR}/config.py
```
