# Real-Time Face Detector

This repository contains training scripts for family of face detectors.

## TBD
Pre-trained models (TABLE WITH quality, input resolution, FPLOS, MPARAMS)

## Training pipeline

1. **Collect dataset**.
2. **Prepare annotation** (annotate or convert existing annotation supported format).
3. Train a model - try both following variants and select the best one:
  * By **training** from scratch or using pre-trained weights - only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to training schedule.
  * By **fine-tuning**. This variant assumes training staring from pre-trained weights with small learning rate that was used in the end of training process of existing snapshot. One should understand that during training model is forgetting about data that was initially trained on (if you don't use it your current training). That's why you should no train you model too long to avoid making model worse than it was before your training has started, especially if you have small dataset.
4. **Evaluate** a model to get quality metrics **in source training framework**.
5. Since PyTorch model is not represented by execution graph we need to:
  1. **Export** model **to ONNX** model.
  2. **Convert** ONNX model **to IR** format that can be efficiently inferred on various Intel platfroms using OpenVINO.
7. **Evaluate** a model in IR fromat to get quality metrics **in OpenVINO**.
8. Run a model in IR format to see how well and how fast it works (**demo**).


### 1. Collect dataset

Download the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) and unpack it to the `data` folder.

### 2. Prepare annotation

Convert downloaded and extracted annotation to MSCOCO format with `face` as the only one class.

* Training annotation
```bash
$ python tools/task_specific/face_detection/wider_to_coco.py \
           data/wider_face_split/wider_face_train_bbx_gt.txt \
           data/WIDER_train/images/ \
           data/train.json
```

* Validation annotation
```bash
$ python tools/task_specific/face_detection/wider_to_coco.py \
           data/wider_face_split/wider_face_val_bbx_gt.txt \
           data/WIDER_val/images/ \
           data/val.json
```

### 3. Training and Fine-tuning
If you would like to start **training** from pre-trained weights do not forget to modify `load_from` path inside configuration file.

If you would like to start **fine-tuning** from pre-trained weights do not forget to modify `resume_from` path inside configuration file as well as increase `total_epochs`. Otherwise training will be ended immideately.
* To train the detector on a single GPU, run in your terminal:
```bash
$ python ../../external/mmdetection/tools/train.py \
           configs/face_detection/<CONFIGURATION_FILE>
```

* To train the detector on multiple GPUs, run in your terminal:
```bash
$ ../../external/mmdetection/tools/dist_train.sh \
           configs/face_detection/<CONFIGURATION_FILE> <GPU_NUM>
```


### 4. Validation

* To dump detection of your model as well as compute MS-COCO metrics run:
```bash
$ python ../../external/mmdetection/tools/test.py \
           configs/face_detection/<CONFIGURATION_FILE> \
           <CHECKPOINT> \
           --out result.pkl \
           --eval bbox
```

* You can also measure WiderFace quality metrics:
  1. Convert `result.pkl` obtained from previous step to WiderFace-friendly output:
  ```bash
  $ python tools/test_out_to_wider_predictions.py \
              configs/face_detection/<CONFIGURATION_FILE> \
              result.pkl \
              <OUTPUT_FOLDER>
  ```
  2. Run WiderFace validation either with
    * [Official Matlab evaluation code](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip)
    * [Python implementation](https://github.com/wondervictor/WiderFace-Evaluation) (anyway you have to download [Official Matlab evaluation code](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip) to get annotation in matlab format). Run from cloned repo following command
    ```bash
    $ python evaluation.py -p <OUTPUT_FOLDER> -g <WIDERFACE_MATLAB_ANNOTATION>
    ```

### 5. Export
* Export to ONNX
  * In most cases you can convert PyTorch\* model to the ONNX\* format by running the `export.py` script:
  ```bash
  $ python ../../external/mmdetection/tools/export.py \
             configs/face_detection/<CONFIGURATION_FILE> \
             <CHECKPOINT> \
             model.onnx
  ```

  * If your model is SSD-like detector you can convert PyTorch\* model to the ONNX\* format by running the `export_ssd.py` script. Note: model exported in such way will produce a bit different results (non-significant in most cases) but it also might be faster that model exported by `export.py`. The `export.py` can export SSD models as well.
  ```bash
  $ python ../../external/mmdetection/tools/export_ssd.py \
             configs/face_detection/<CONFIGURATION_FILE> \
             <CHECKPOINT> \
             model.onnx
  ```
  2. TBD. Convert ONNX model to the OpenVINO™ format with the Model Optimizer with the command below:
  ```bash
  python ../../external/mmdetection/tools/convert_to_ir.py \
            configs/face_detection/<CONFIGURATION_FILE> \
            model.onnx \
            <EXPORT_FOLDER>
  ```
    This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
    (FP32). The obtained model expects **normalized image** in planar BGR format.

### 6. Demo

Please refer to ???WHICH DEMO??? [OpenModelZoo](https://github.com/opencv/open_model_zoo).


## Other
### Theoretical computational complexity estimation

To get per-layer computational complexity estimations, run the following command:
```bash
$ python ../../external/mmdetection/tools/get_flops.py \
         configs/face_detection/<CONFIGURATION_FILE>
```
