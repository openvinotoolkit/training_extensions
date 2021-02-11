# Image classification
## Description of the models
### Performance results pivot table

| Model Name | Complexity (GFLOPs) | Size (Mp) | Top-1 on ImageNet (%) | Links |
| ---        | ---                 | ---       | ---                   | ---                   |
| MobilenetV2 1.0 | 0.66           | 3.50      | 73.0                  | [model template](./mobilenet_v2_w1/template.yaml)

| Model Name | Complexity (GFLOPs) | Size (Mp) | Latency (ms) | FPS | AVG mAP (%) | AVG Top-1 (%) | AVG Top-5 (%) | Links |
| ---        | ---        | ---        | ---        | ---        | ---        | ---        | ---        | ---        |
| EfficientNet b0 | 0.76 | 4.14 | 9.7 | 447.51 | 92.267 | 90.156 | 98.283 | [model template](./efficient_b0/template.yaml) |
| MobilenetV3 large x1.0 | 0.44 | 4.33 | 5.58 | 792.22 | 83.46	| 87.06 | 96.42 | [model template](./mobilenet_v3_large_1/template.yaml) |
| MobilenetV3 large x0.75 | 0.308 | 2.84 | 4.48 | 985.4 | 87.59 |	87.59	| 97.17 | [model template](./mobilenet_v3_large_075/template.yaml) |
| MobilenetV3 small x1.0 | 0.112 | 1.56 | 2.79 | 1599.39 | 82.84 | 84.47 | 95.88 | [model template](./mobilenet_v3_small/template.yaml) |

All of the above metrics were obtained on eleven different datasets, on which an extensive number of research had been made.

Performance mesured on CPU: Intel(R) Core(TM) i9-9820X CPU @ 3.30GHz,  OpenVINO 2021.2​, batch size=1, input resolution is 224 × 224, inference precision is FP32
### Datasets and information about metrics

The following datasets were used for experiments:
* [Describable Textures (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)<sup>1</sup>
* [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)<sup>1</sup>
* [Oxford 102 Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN (w/o additional data)](http://ufldl.stanford.edu/housenumbers/)
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [FOOD101](https://www.kaggle.com/dansbecker/food-101)<sup>1</sup>
* [SUN397](https://vision.princeton.edu/projects/2010/SUN/)<sup>1</sup>
* [Birdsnap](http://thomasberg.org/)<sup>1</sup>
* [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

<sup>1</sup> these dataset have custom splits (random split 80% - train, 20% - val) and cannot be compared straightforward with other research results

For additional information about perfomance on each dataset in comparison with baseline you can refer to this [spreadsheet]().
### Training recipe

Baselines:
* softmax loss
* most appropriate average learning rate for all datasets (0.016)
* cosine scheduler
* basic augmentations
* SGD with momentum optimizer

MobilenetV3:
* Mutual learning approach
* Softmax loss for the main model, AM_softmax for the auxiliary model
* Learning rate found by LR Finder
* Reduce on platue scheduler which allow to get rid of epochs search
* Augmix pipeline for augmentations
* Sharpness aware minimization optimizer

EfficientNet_b0:
* Softmax loss
* Learning rate found by LR Finder
* Reduce on platue scheduler which allow to get rid of epochs search
* Augmix pipeline for augmentations + FMIX augmentation
* Sharpness aware minimization optimizer

All of the models were initially trained on [ImageNet](http://image-net.org/challenges/LSVRC/2012/), but can be trained from scratch or fine tuned to classify arbitrary images.
## Training pipeline

### 0. Change a directory in your terminal to image_classification.

```bash
cd <training_extensions>/pytorch_toolkit/image_classification
```

### 1. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/custom-classification/mobilenet_v3_large_1/template.yaml`
export WORK_DIR=/tmp/my_model
python ../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR} --do-not-load-snapshot
```

### 2. Prepare data

The training script assumes the data for classification is divided by folders in such a way when
each class has its own folder. The script automatically computes number of subfolders in the train
dataset directory and assumes each directory to represent one class. Indexes of classes are assigned
according to alphabetically sorted list of folders.

An example of the directory structure:

```
DATA_DIR
├── train
│   ├── Class1
|   |   └── <train images that belong to class 1>
│   ├── Class2
|   |   └── <train images that belong to class 2>
│   |── ....
│   └── ClassN
|       └── <train images that belong to class N>
│
└── val
    ├── Class1
    |   └── <val images that belong to class 1>
    ├── Class2
    |   └── <val images that belong to class 2>
    └── ....
    └── ClassN
        └── <val images that belong to class N>
```

After the data was arranged, export the variables required for launching training and evaluation scripts:

```bash
export TRAIN_DATA_ROOT=${DATA_DIR}/train
export VAL_DATA_ROOT=${DATA_DIR}/val
export TEST_DATA_ROOT=${DATA_DIR}/val
```

### 3. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 4. Training and Fine-tuning

Try both following variants and select the best one:

* **Training from scratch** from scratch or pre-trained weights. Only if you have a lot of data, let's say tens of thousands or even more images. This variant assumes long training process starting from big values of learning rate and eventually decreasing it according to a training schedule.
* **Fine-tuning** from pre-trained weights. If the dataset is not big enough, then the model tends to overfit quickly, forgetting about the data that was used for pre-training and reducing the generalization ability of the final model. Hence, small starting learning rate and short training schedule are recommended.

```bash
python train.py \
   --train-ann-files '' \
   --train-data-roots ${TRAIN_DATA_ROOT} \
   --val-ann-files '' \
   --val-data-roots ${VAL_DATA_ROOT} \
   --save-checkpoints-to ${WORK_DIR}/outputs
```

> **NOTE**: It's recommended during fine-tuning to decrease the `--base-learning-rate` parameter compared with default value (see `${MODEL_TEMPLATE}`) to prevent from forgetting during the first iterations.

Also you can use parameters such as `--epochs`, `--batch-size`, `--gpu-num`, `--base-learning-rate`, otherwise default values will be loaded from `${MODEL_TEMPLATE}`.

### 5. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute mean accuracy metric run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/model/model.pth.tar-150 \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 6. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.
