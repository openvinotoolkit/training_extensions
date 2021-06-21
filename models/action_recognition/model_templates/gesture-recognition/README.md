# Gesture Recognition

Models that are able to recognize gestures from live video stream on CPU.

* MS-ASL-100 gesture set (continuous scenario)

  | Model Name                        | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy  | Links                                                                                                                                                                                                                                 | GPU_NUM |
  | --------------------------------- | ------------------- | --------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
  | s3d-rgb-mobilenet-v3-stream-msasl | 6.66	              | 4.133     | 84.7%           | [model template](s3d-rgb-mobilenet-v3-stream-msasl/template.yaml), [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/asl/s3d-mobilenetv3-large-statt-msasl1000.pth)                                       | 2       |

* Jester-27 gesture set (continuous scenario)

  | Model Name                         | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy | Links                                                                                                                                                                                                                                 | GPU_NUM |
  | ---------------------------------- | ------------------- | --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
  | s3d-rgb-mobilenet-v3-stream-jester | 4.23	               | 4.133     | 93.58%         | [model template](s3d-rgb-mobilenet-v3-stream-jester/template.yaml), [snapshot](https://docs.google.com/uc?export=download&id=1lDm2qOxMRyXZW6y7owlQBv8SGvGIKpcX)                                                                       | 4       |

* Common-Sign-Language-12 gesture set (continuous scenario)

  | Model Name                         | Complexity (GFLOPs) | Size (Mp) | Top-1 accuracy | Links                                                                                                                                                                                                                                 | GPU_NUM |
  | ---------------------------------- | ------------------- | --------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
  | s3d-rgb-mobilenet-v3-stream-csl    | 4.23                | 4.113     | 98.00%         | [model template](s3d-rgb-mobilenet-v3-stream-csl/template.yaml), [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/common_sign_language/s3d-mobilenetv3-large-common_sign_language.pth) | 2       |

## Usage

Steps `1`-`2` help to setup working environment and download a pre-trained model.
Steps `3.a`-`3.c` demonstrate how the pre-trained model can be exported to OpenVINO compatible format and run as a live-demo.
If you are unsatisfied by the model quality, steps `4.a`-`4.c` help you to prepare datasets, evaluate pre-trained model and run fine-tuning.
You can repeat steps `4.b` - `4.c` until you get acceptable quality metrics values on your data, then you can re-export model and run demo again (Steps `3.a`-`3.c`).

### 1. Change a directory in your terminal to domain directory

```bash
cd models/action_recognition
```
If you have not created virtual environment yet:
```bash
./init_venv.sh
```
Activate virtual environment:
```bash
source venv/bin/activate
```

### 2. Select a model template file and instantiate it in some directory

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/gesture-recognition/s3d-rgb-mobilenet-v3-stream-msasl/template.yaml`
export WORK_DIR=/tmp/my-$(basename $(dirname $MODEL_TEMPLATE))
export SNAPSHOT=snapshot.pth
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR}
```

### 3. Try a pre-trained model

#### a. Change current directory to directory where the model template has been instantiated

```bash
cd ${WORK_DIR}
```
#### b. Export pre-trained PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${SNAPSHOT} \
   --save-model-to export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

#### c. Run demo with exported model

You need to pass a path to `model.xml` file and the index of your web cam. Also a video file probably can be used as an input (-i) for the demo, please refer to documentation in [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) repo.

```bash
python ${OMZ_DIR}/tools/downloader/downloader.py \
  --name person-detection-asl-0001 \
  --precisions FP32
python ${OMZ_DIR}/demos/gesture_recognition_demo/python/gesture_recognition_demo.py \
  -m_a export/model.xml \
  -m_d intel/person-detection-asl-0001/FP32/person-detection-asl-0001.xml \
  -i 0 \
  -c ${OMZ_DIR}/data/dataset_classes/msasl100.json
```

### 4. Fine-tune

#### a. Prepare dataset

Prepare one of the listed below datasets for training or collect and annotate your own:
* To prepare MS-ASL data follow instructions: [DATA_MSASL.md](./DATA_MSASL.md).
* To prepare JESTER data follow instructions: [DATA_JESTER.md](./DATA_JESTER.md).

Set some environment variables:
```bash
export ADD_EPOCHS=1
export EPOCHS_NUM=$((`cat ${MODEL_TEMPLATE} | grep epochs | tr -dc '0-9'` + ${ADD_EPOCHS}))
```

#### b. Evaluate

```bash
python eval.py \
   --load-weights ${SNAPSHOT} \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to metrics.yaml
```

If you would like to evaluate exported model, you need to pass `export/model.bin` instead of passing `${SNAPSHOT}` .

#### c. Fine-tune or train from scratch

Try both following variants and select the best one:

   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

   * If you would like to start **training** from pre-trained weights use `--load-weights` pararmeter. Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

      ```bash
         python train.py \
         --load-weights ${SNAPSHOT} \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_DATA_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_DATA_ROOT} \
         --save-checkpoints-to outputs \
      && export SNAPSHOT=outputs/latest.pth

   * If you would like to start **fine-tuning** from your pre-trained weights use `--resume-from` parameter and value of `--epochs` have to exceed the value stored inside `${MODEL_TEMPLATE}` file, otherwise training will be ended immediately. Here we add `1` additional epoch.

     Important: the `--resume-from` does not work with provided pre-trained weights, but one can resume its own training.

      ```bash
         python train.py \
         --resume-from ${SNAPSHOT} \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_DATA_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_DATA_ROOT} \
         --save-checkpoints-to outputs \
         --epochs ${EPOCHS_NUM} \
      && export SNAPSHOT=outputs/latest.pth \
      && export EPOCHS_NUM=$((${EPOCHS_NUM} + ${ADD_EPOCHS}))
      ```

As soon as training is completed, it is worth to re-evaluate trained model on test set (see Step 4.b).
