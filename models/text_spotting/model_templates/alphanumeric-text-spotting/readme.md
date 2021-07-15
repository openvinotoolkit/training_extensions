# Alphanumeric Text Spotting

Models that are able to detect and recognize alphanumeric text (numbers and letters of English alphabet).

| Model Name                    | Complexity (GFLOPs) | Size (Mp) | Detection F1-score (ICDAR'15) | Word Spotting (ICDAR'15, N*) | E2E Recognition (ICDAR'15, N*) |  Word Spotting (ICDAR'15, G**) | E2E Recognition (ICDAR'15, G**) | Links                                                                                                                                                                                                                           | GPU_NUM |
| ----------------------------- | ------------------- | --------- | ----------------------------- | ---------------------------- | ------------------------------ | ------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| text-spotting-0005 ([YAMTS](https://arxiv.org/abs/2106.12326)) @ 1280x768 | 190.5               |  27.76    |  88.87% @ 0.65,0.45             | 71.29% @ 0.65,0.45             | 68.55% @ 0.65,0.45               | 76.63% @ 0.65,0.15              | 74.12% @ 0.65,0.15               | [model template](./text-spotting-0005/template.yaml), [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/text_spotting/alphanumeric_text_spotting/text_spotting_0005/epoch_25.pth) | 4       |

Metric value has following format: <metric_value> @ <detection_threshold>,<recognition_threshold>.

*N - no lexicon was used.

**G - [generic lexicon](https://drive.google.com/file/d/1XQsikiNY7ILgZvmvOeUf9oPDG4fTp0zs/view?usp=sharing) was used.

## Usage

Steps `1`-`2` help to setup working environment and download a pre-trained model.
Steps `3.a`-`3.c` demonstrate how the pre-trained model can be exported to OpenVINO compatible format and run as a live-demo.
If you are unsatisfied by the model quality, steps `4.a`-`4.c` help you to prepare datasets, evaluate pre-trained model and run fine-tuning.
You can repeat steps `4.b` - `4.c` until you get acceptable quality metrics values on your data, then you can re-export model and run demo again (Steps `3.a`-`3.c`).

### 1. Change a directory in your terminal to domain directory

```bash
cd models/text_spotting
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
export MODEL_TEMPLATE=`realpath ./model_templates/alphanumeric-text-spotting/text-spotting-0005/template.yaml`
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

This produces models `model*.xml` and weights `model*.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.

#### c. Run demo with exported model

You need to pass a path to `model*.xml` files and video device node (e.g. /dev/video0) of your web cam. Also an image or a video file probably can be used as an input (-i) for the demo, please refer to documentation in [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) repo.

```bash
python ${OMZ_DIR}/demos/text_spotting_demo/python/text_spotting_demo.py \
   -m_m export/model.xml \
   -m_te export/model_text_recognition_head_encoder.xml \
   -m_td export/model_text_recognition_head_decoder.xml \
   -i /dev/video0
```

### 4. Fine-tune

#### a. Prepare dataset

In this toy example we use same images as training, validation and test subsets, but we strictly recommend not to use the same data for training, validation and test. This particular example is for demonstration of model quality growth on particular dataset during fine-tuning only. See more about dataset split [here](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets).

In order to train a model that would be quite similar in terms of quality to existing pre-trained model one can prepare [publicly-available datasets](datasets.md) for training. One can also use its own preliminary annotated dataset. Annotation can be created using [CVAT](https://github.com/openvinotoolkit/cvat) as we did in this toy example.

Training images are stored in `${TRAIN_IMG_ROOT}` together with `${TRAIN_ANN_FILE}` annotation file and validation images are stored in `${VAL_IMG_ROOT}` together with `${VAL_ANN_FILE}` annotation file.

```bash
export ADD_EPOCHS=1
export EPOCHS_NUM=$((`cat ${MODEL_TEMPLATE} | grep epochs | tr -dc '0-9'` + ${ADD_EPOCHS}))
export TRAIN_ANN_FILE=${OTE_DIR}/data/horizontal_text_detection/annotation.json
export TRAIN_IMG_ROOT=${OTE_DIR}/data/horizontal_text_detection/
export VAL_ANN_FILE=${TRAIN_ANN_FILE}
export VAL_IMG_ROOT=${TRAIN_IMG_ROOT}
export TEST_ANN_FILE=${TRAIN_ANN_FILE}
export TEST_IMG_ROOT=${TRAIN_IMG_ROOT}
```

#### b. Evaluate

```bash
python eval.py \
   --load-weights ${SNAPSHOT} \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_IMG_ROOT} \
   --save-metrics-to metrics.yaml
```

If you would like to evaluate exported model, you need to pass `export/model.bin` instead of passing `${SNAPSHOT}` .

#### c. Fine-tune or train from scratch

Try both following variants and select the best one:

   * **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.
   * **Training** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.

   * If you would like to start **fine-tuning** from pre-trained weights use `--resume-from` parameter and value of `--epochs` have to exceed the value stored inside `${MODEL_TEMPLATE}` file, otherwise training will be ended immediately. Here we add `1` additional epoch.

      ```bash
      python train.py \
         --resume-from ${SNAPSHOT} \
         --train-ann-files ${TRAIN_ANN_FILE} \
         --train-data-roots ${TRAIN_IMG_ROOT} \
         --val-ann-files ${VAL_ANN_FILE} \
         --val-data-roots ${VAL_IMG_ROOT} \
         --save-checkpoints-to outputs \
         --epochs ${EPOCHS_NUM} \
      && export SNAPSHOT=outputs/latest.pth \
      && export EPOCHS_NUM=$((${EPOCHS_NUM} + ${ADD_EPOCHS}))
      ```

   * If you would like to start **training** from pre-trained weights use `--load-weights` pararmeter instead of `--resume-from`. Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

As soon as training is completed, it is worth to re-evaluate trained model on test set (see Step 4.b).
