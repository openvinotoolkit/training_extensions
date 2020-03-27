# Person Vehicle Bike Detector

The crossroad-detection network model provides detection of three class objects: vehicle, pedestrian, non-vehicle (like bikes). This detector was trained on the data from crossroad cameras.
| Model Name                  | Complexity (GFLOPs) | Size (Mp) | Mean Average Precision (mAP) | Links                                                                        |
| --------------------------- | ------------------- | --------- | ------------- | ---------------------------------------------------------------------------- |
| person-vehicle-bike-detection-crossroad-1016  | 3.560               | 	2.887    | 62.55%          | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/person_vehicle_bike_sd512_mb2_clustered_epoch_21.pth), [configuration file](./configs/person-vehicle-bike-detection-crossroad-1016.py) |

Average Precision (AP) is defined as an area under the precision/recall curve.

## Training pipeline

### 1. Select a training configuration file and get pre-trained snapshot if available. Please see the table above.

```bash
$ export CONFIGURATION_FILE=./configs/person-vehicle-bike-detection-crossroad-1016.py
```

### 2. Collect dataset

You can train a model on existing toy dataset `openvino_training_extensions/data/airport`. Obviously such dataset is not sufficient for training good enough model.

### 3. Prepare annotation

The existing toy dataset has annotation in the Common Objects in Context (COCO) and mmdetection CustomDataset format.

### 4. Training and Fine-tuning
Try both following variants and select the best one:
   * **Training** from scratch or pre-trained weights -- only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

If you would like to start **training** from pre-trained weights do not forget to modify `load_from` path inside configuration file.

If you would like to start **fine-tuning** from pre-trained weights do not forget to modify `resume_from` path inside configuration file as well as increase `total_epochs`. Otherwise training will be ended immideately.
* To train the detector on a single GPU, run in your terminal:
   ```bash
   $ python ../../external/mmdetection/tools/train.py \
            $CONFIGURATION_FILE
   ```

* To train the detector on multiple GPUs, run in your terminal:
   ```bash
   $ ../../external/mmdetection/tools/dist_train.sh \
            $CONFIGURATION_FILE \
            <GPU_NUM>
   ```


### 5. Validation

* To dump detection of your model as well as compute MS-COCO metrics run:
   ```bash
   $ python ../../external/mmdetection/tools/test.py \
            $CONFIGURATION_FILE \
            <CHECKPOINT> \
            --out result.pkl \
            --eval bbox
   ```

### 6. Export
* Export to ONNX
  * In most cases you can convert PyTorch\* model to the ONNX\* format by running the `export.py` script:
     ```bash
     $ python ../../external/mmdetection/tools/export.py \
              $CONFIGURATION_FILE \
              <CHECKPOINT> \
              model.onnx
     ```

  * If your model is SSD-like detector you can convert PyTorch\* model to the ONNX\* format by running the `export_ssd.py` script. Note: model exported in such way will produce a bit different results (non-significant in most cases) but it also might be faster that model exported by `export.py`. The `export.py` can export SSD models as well.
     ```bash
     $ python ../../external/mmdetection/tools/export_ssd.py \
              $CONFIGURATION_FILE \
              <CHECKPOINT> \
              model.onnx
     ```
  2. Convert ONNX model to the OpenVINO™ format with the Model Optimizer with the command below:
     ```bash
     python ../../external/mmdetection/tools/convert_to_ir.py \
            $CONFIGURATION_FILE \
            model.onnx \
            <EXPORT_FOLDER>
     ```
    This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
    (FP32). The obtained model expects **normalized image** in planar BGR format.

### 7. Validataion of IR

Instead running of `test.py` you need to run `test_exported.py` and then repeat steps listed in [Validation paragraph](#5-validation).

If you exported model using `export_ssd.py` you need to add `--with_detection_output` option, otherwise you don't need to use such flag. If you converted the exported model using `convert_to_ir.py` then the model expects **normalized image** in planar BGR format, so you need to disable data pre-processing by adding `--do_not_normalize`.
   ```bash
   python ../../external/mmdetection/tools/test_exported.py  \
          $CONFIGURATION_FILE \
          <EXPORT_FOLDER>/model.xml \
          --with_detection_output \
          --do_not_normalize \
          --out results.pkl \
          --eval bbox
   ```

### 8. Demo

To see how the converted model works using OpenVINO you need to run `test_exported.py` with `--show` option.
   ```bash
   python ../../external/mmdetection/tools/test_exported.py  \
          $CONFIGURATION_FILE \
          <EXPORT_FOLDER>/model.xml \
          --with_detection_output \
          --do_not_normalize \
          --show
   ```


## Other
### Theoretical computational complexity estimation

To get per-layer computational complexity estimations, run the following command:
   ```bash
   $ python ../../external/mmdetection/tools/get_flops.py \
            $CONFIGURATION_FILE
   ```
