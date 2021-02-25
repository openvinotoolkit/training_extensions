# Super Resolution for Scanned Text Images

The tiny model to upscale scanned images with text. The model uses the `ConvTranspose2d` layer instead of `PixelShuffle`, so the
model can be launched on GPU and MYRIAD devices and Inference Engine support `reshape` function.

## Train and Evaluation

### Prepare Dataset

Create two directories for train and test images. Train images may have any resolution higher than the `path_size`.
Validation images should have the resolution like the `path_size`.

```
./data
├── train
│   ├── 000000.png
│   ...
└── val
    ├── 000000.png
    ...
```

>**NOTE**: Image should be in the gray scale format and contain only black (0) and white (255) pixels.

> **TIP**: It is better to use cropped images like 500x500, because large resolution dramatically increases the time to read images.


### Training

Use the `tools/train.py` script to start the training process:
```
python3 tools/train.py --config configs/text_scale3.yaml
```

To start from the pretrained [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/super_resolution/text_super_resolution.tar.gz), set `init_checkpoint` in config.


### Test

Use the `tools/test.py` script to evaluate the trained model:

```
python3 tools/test.py --test_data_path PATH_TO_TEST_DATA \
    --models_path PATH_TO_MODELS_PATH \
    --exp_name EXPERIMENT_NAME
```

## Export to OpenVINO™

```
python3 tools/export.py --models_path PATH_TO_MODELS_PATH \
    --exp_name EXPERIMENT_NAME \
    --input_size 200 200 \
    --data_type FP32
```

## Demo

### For the Latest Checkpoint

```
python3 tools/text/infer.py --model PATH_TO_CHECKPOINT IMAGE_PATH
```

### For Intermediate Representation (IR)

```
python3 tools/text/infer_ie.py --model <PATH_TO_IR_XML> \
    --device CPU \
    image_path
```
