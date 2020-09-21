# Face Detection

Models that are able to detect faces.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | AP for faces > 64x64 (%) | WiderFace Easy (%) | WiderFace Medium (%) | WiderFace Hard (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face-detection-0200 | 0.82 | 1.83 | 16.0 | 86.743 | 82.917 | 76.198 | 41.443 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth), [configuration file](./face-detection-0200/config.py) | 2 |
| face-detection-0202 | 1.84 | 1.83 | 20.3 | 91.938 | 89.382 | 83.919 | 50.189 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0202.pth), [configuration file](./face-detection-0202/config.py) | 2 |
| face-detection-0204 | 2.52 | 1.83 | 21.4 | 92.888 | 90.453 | 85.448 | 52.091 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0204.pth), [configuration file](./face-detection-0204/config.py) | 4 |
| face-detection-0205 | 2.94 | 2.02 | 21.6 | 93.566 | 92.032 | 86.717 | 54.055 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0205.pth), [configuration file](./face-detection-0205/config.py) | 4 |
| face-detection-0206 | 340.06 | 63.79 | 34.2 | 94.274 | 94.281 | 93.207 | 84.439 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0206.pth), [configuration file](./face-detection-0206/config.py) | 8 |
| face-detection-0207 | 1.04 | 0.81 | 17.2 | 88.17 | 84.406 | 76.748 | 43.452 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0207.pth), [configuration file](./face-detection-0207/config.py) | 1 | 

## Training pipeline

### 0. Change a directory in your terminal to object_detection.

```bash
cd <training_extensions>/pytorch_toolkit/object_detection/model_templates
```

### 1. Select a training configuration file and get pre-trained snapshot if available. Please see the table above.

```bash
export MODEL_NAME=face-detection-0100
export CONFIGURATION_FILE=./face-detection/$MODEL_NAME/config.py
```

### 2. Collect dataset

Download the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) and unpack it to the `data` folder.

### 3. Prepare annotation

Convert downloaded and extracted annotation to MSCOCO format with `face` as the only one class.

* Training annotation

   ```bash
   python face-detection/tools/wider_to_coco.py \
            data/wider_face_split/wider_face_train_bbx_gt.txt \
            data/WIDER_train/images/ \
            data/train.json
   ```

* Validation annotation

   ```bash
   python face-detection/tools/wider_to_coco.py \
            data/wider_face_split/wider_face_val_bbx_gt.txt \
            data/WIDER_val/images/ \
            data/val.json
   ```

### 4. Training and Fine-tuning

Try both following variants and select the best one:

   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

If you would like to start **training** from pre-trained weights do not forget to modify `load_from` path inside configuration file.

If you would like to start **fine-tuning** from pre-trained weights do not forget to modify `resume_from` path inside configuration file as well as increase `total_epochs`. Otherwise training will be ended immideately.

* To train the detector on a single GPU, run in your terminal:

   ```bash
   python ../../../external/mmdetection/tools/train.py \
            $CONFIGURATION_FILE
   ```

* To train the detector on multiple GPUs, run in your terminal:

   ```bash
   ../../../external/mmdetection/tools/dist_train.sh \
            $CONFIGURATION_FILE \
            <GPU_NUM>
   ```
* To train the detector on multiple GPUs and to perform quality metrics estimation as soon as training is finished, run in your terminal

   ```bash
   python face-detection/tools/train_and_eval.py \
            $CONFIGURATION_FILE \
            <GPU_NUM>
   ```

   If you have WiderFace dataset downloaded you also can specify `--wider_dir` parameter where `WIDER_val.zip` file is stored (so that <WIDER_FACE>/WIDER_val.zip)

   ```bash
   python face-detection/tools/train_and_eval.py \
            $CONFIGURATION_FILE \
            <GPU_NUM> \
            --wider_dir <WIDER_FACE_DIR>
   ```

### 5. Validation

* To dump detection of your model as well as compute MS-COCO metrics run:

   ```bash
   python ../../../external/mmdetection/tools/test.py \
            $CONFIGURATION_FILE \
            <CHECKPOINT> \
            --out result.pkl \
            --eval bbox
   ```

* You can also measure WiderFace quality metrics:

  1. Convert `result.pkl` obtained from previous step to WiderFace-friendly output:

     ```bash
     python face-detection/tools/test_out_to_wider_predictions.py \
              $CONFIGURATION_FILE \
              result.pkl \
              <OUTPUT_FOLDER>
     ```

  2. Run WiderFace validation either with

     * [Official Matlab evaluation code](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)
     * [Python implementation](https://github.com/wondervictor/WiderFace-Evaluation) (anyway you have to download [Official Matlab evaluation code](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip) to get annotation in matlab format). Run from cloned repo following command

        ```bash
        python evaluation.py -p <OUTPUT_FOLDER> -g <WIDERFACE_MATLAB_ANNOTATION>
        ```

### 6. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python ../../../external/mmdetection/tools/export.py \
      $CONFIGURATION_FILE \
      <CHECKPOINT> \
      <EXPORT_FOLDER> \
      openvino
```

This produces model `$MODEL_NAME.xml` and weights `$MODEL_NAME.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

For SSD networks an alternative OpenVINO™ representation is possible.
To opt for it use extra `--alt_ssd_export` key to the `export.py` script.
SSD model exported in such way will produce a bit different results (non-significant in most cases),
but it also might be faster than the default one. As a rule SSD models in [Open Model Zoo](https://github.com/opencv/open_model_zoo/) are exported using this option.

### 7. Validation of IR

Instead of running `test.py` you need to run `test_exported.py` and then repeat steps listed in [Validation paragraph](#5-validation).

```bash
python ../../../external/mmdetection/tools/test_exported.py  \
      $CONFIGURATION_FILE \
      <EXPORT_FOLDER>/$MODEL_NAME.xml \
      --out results.pkl \
      --eval bbox
```

### 8. Demo

To see how the converted model works using OpenVINO you need to run `test_exported.py` with `--show` option.

```bash
python ../../../external/mmdetection/tools/test_exported.py  \
      $CONFIGURATION_FILE \
      <EXPORT_FOLDER>/$MODEL_NAME.xml \
      --show
```

## Other

### Theoretical computational complexity estimation

To get per-layer computational complexity estimations, run the following command:

```bash
python ../../../external/mmdetection/tools/get_flops.py \
       $CONFIGURATION_FILE
```

