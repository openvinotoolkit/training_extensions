# Image classification

Performance results pivot table:

| Model Name | Complexity (GFLOPs) | Size (Mp) | AVG mAP (%) | AVG Top-1 (%) | AVG Top-5 (%) | Links |
| ---        | ---        | ---        | ---        | ---        | ---        | ---        |
| EfficientNet b0 | 0.76 | 4.14 | 92.75 | 89.14	| 97.79 | [imagenet snapshot](https://drive.google.com/uc?export=download&id=1AyOTxxuJPrX9H3whcgaQ_BRRDiZmbqt0), [model template](./efficientnet_b0/template.yaml) |
| MobilenetV3 large x1.0 | 0.44 | 4.33 | 91.98 | 88.30 | 97.35 | [imagenet snapshot](https://drive.google.com/uc?export=download&id=1pJ86SPmRrmnGhrNGyXV8FMKSu550OnV1), [model template](./mobilenet_v3_large_1/template.yaml) |
| MobilenetV3 large x0.75 | 0.308 | 2.84 | 91.14 | 87.60	| 96.97 | [imagenet snapshot](https://drive.google.com/uc?export=download&id=1s9Z43yyL281QnDMRmBRbBiizzAqVvM_0), [model template](./mobilenet_v3_large_075/template.yaml) |
| MobilenetV3 small x1.0 | 0.112 | 1.56 | 87.81	| 84.99	| 96.15 | [imagenet snapshot](https://drive.google.com/uc?export=download&id=1q3xgROzhFWCiQHPvKldV4S7HR3N2Fne1), [model template](./mobilenet_v3_small/template.yaml) |


All of the above metrics were obtained on eleven different datasets, on which an extensive amount of research has been made. To provide a generalized performance metric, we averaged the metrics across all datasets. For additional information about performance on each dataset in comparison with baseline, you can refer to this [spreadsheet](https://docs.google.com/spreadsheets/d/1CV3be-VydEHvWS6GMPduBQBjl46uLq80_GtkeUhsuVg/edit#gid=0).

The following datasets were used in experiments:
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

<sup>1</sup> these datasets have custom splits (random stratified splits) and cannot be compared straightforwardly with other research results

Training recipes:

We pretrained all models with imagenet weights and fine-tuned on specific tasks without freezing any layers.

The following parameters and techniques were used for training:

Baselines:
* softmax loss
* most appropriate average learning rate for all datasets (0.013 for MobileNetV3 and 0.003 for EfficientNet)
* cosine scheduler
* basic augmentations
* SGD with momentum optimizer

MobilenetV3:
* [Mutual learning](https://www.semanticscholar.org/paper/Deep-Mutual-Learning-Zhang-Xiang/f06a12928307e17b1aff2b9f4a6c11791f19b6a7) approach
* Softmax loss for the main model, [Additive Margin softmax](https://www.semanticscholar.org/paper/Additive-Margin-Softmax-for-Face-Verification-Wang-Cheng/9fc17fa5708584fa848164461f82a69e97f6ed69) for the auxiliary model
* Learning rate found by LR Finder
* Reduce on plateau scheduler which allows getting rid of epochs search
* Augmix pipeline for augmentations
* [Sharpness aware minimization optimizer](https://www.semanticscholar.org/paper/Sharpness-Aware-Minimization-for-Efficiently-For%C3%AAt-Kleiner/bc52ab18399aaaf6b88c22ebc6e4a3caa99a2323)
* [No bias decay method](https://www.semanticscholar.org/paper/Bag-of-Tricks-for-Image-Classification-with-Neural-He-Zhang/29309743870c825f9645a4803af727402462e513)
* Exponentially Moving Average (EMA) of the weights

EfficientNet_b0:
* [Additive Margin softmax] loss (https://www.semanticscholar.org/paper/Additive-Margin-Softmax-for-Face-Verification-Wang-Cheng/9fc17fa5708584fa848164461f82a69e97f6ed69)
* Learning rate found by LR Finder
* Reduce on plateau scheduler which allows getting rid of epochs search
* [Augmix](https://www.semanticscholar.org/paper/AugMix%3A-A-Simple-Data-Processing-Method-to-Improve-Hendrycks-Mu/f3a93e20a12532b5493825b921a0f0132736f4ec) pipeline for augmentations
* [Sharpness aware minimization optimizer](https://www.semanticscholar.org/paper/Sharpness-Aware-Minimization-for-Efficiently-For%C3%AAt-Kleiner/bc52ab18399aaaf6b88c22ebc6e4a3caa99a2323)
* [No bias decay method](https://www.semanticscholar.org/paper/Bag-of-Tricks-for-Image-Classification-with-Neural-He-Zhang/29309743870c825f9645a4803af727402462e513)
* Exponentially Moving Average (EMA) of the weights

All of the models were initially trained on [ImageNet](http://image-net.org/challenges/LSVRC/2012/), but can be trained from scratch or fine tuned to classify arbitrary images.

Information about LR Finder:

There are three algorithms for estimating an optimal learning rate available: [Optuna's](https://arxiv.org/abs/1907.10902) grid search, TPE and by fast.ai approach imported from [torch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py) with some modifications.

Recommended parameters for the automatic mode in case of fine-tuning:

Mobilenet_v3 backbones:
* min_lr = 0.005
* max_lr = 0.03
* warmup = 1

efficientnet_b0:
* min_lr = 0.001
* max_lr = 0.01
* warmup = 1

The decision will be made automatically by the steepest gradient of the loss function changing.

Also, you can stop after searching learning rate (`stop_after=True`), build a graphic of the loss function (`path_to_savefig: 'some/path/to/figure'`), and make your own decision about choosing learning rate.
## Training pipeline

### 1. Change a directory in your terminal to image_classification.

```bash
cd models/image_classification
```
If You have not created virtual environment yet:
```bash
./init_venv.sh
```
Activate virtual environment:
```bash
source venv/bin/activate
```

### 2. Select a model template file and instantiate it in some directory.

```bash
export MODEL_TEMPLATE=`realpath ./model_templates/custom-classification/mobilenet_v3_large_1/template.yaml`
export WORK_DIR=/tmp/my_model
python ../../tools/instantiate_template.py ${MODEL_TEMPLATE} ${WORK_DIR} --do-not-load-snapshot
```

### 3. Prepare data

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

### 4. Change current directory to directory where the model template has been instantiated.

```bash
cd ${WORK_DIR}
```

### 5. Training and Fine-tuning

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

### 6. Evaluation

Evaluation procedure allows us to get quality metrics values and complexity numbers such as number of parameters and FLOPs.

To compute mean accuracy metric run:

```bash
python eval.py \
   --load-weights ${WORK_DIR}/outputs/model/model.pth.tar-150 \
   --test-ann-files ${TEST_ANN_FILE} \
   --test-data-roots ${TEST_DATA_ROOT} \
   --save-metrics-to ${WORK_DIR}/metrics.yaml
```

### 7. Export PyTorch\* model to the OpenVINO™ format

To convert PyTorch\* model to the OpenVINO™ IR format run the `export.py` script:

```bash
python export.py \
   --load-weights ${WORK_DIR}/outputs/latest.pth \
   --save-model-to ${WORK_DIR}/export
```

This produces model `model.xml` and weights `model.bin` in single-precision floating-point format
(FP32). The obtained model expects **normalized image** in planar BGR format.


### 8. Optimization

The models can be optimized -- compressed by [NNCF](https://github.com/openvinotoolkit/nncf) framework.

To use NNCF to compress an image classification model, you should go to the root folder of this git repository
and install compression requirements in your virtual environment by the command
```bash
pip install -r external/deep-object-reid/compression_requirements.txt
```

At the moment, only one compression method is supported for image classification models:
[int8 quantization](https://github.com/openvinotoolkit/nncf/blob/develop/docs/compression_algorithms/Quantization.md).

To compress the model, 'compress.py' script should be used.

Please, note that NNCF framework requires a dataset for compression, since it makes several steps of fine-tuning after
compression to restore the quality of the model, so the command line parameters of the script `compress.py` are closer
to the command line parameter of the training script for fine-tuning scenario from the section 5 stated above:
```
python compress.py \
   --load-weights ${SNAPSHOT} \
   --train-ann-files '' \
   --train-data-roots ${TRAIN_DATA_ROOT} \
   --val-ann-files '' \
   --val-data-roots ${VAL_DATA_ROOT} \
   --save-checkpoints-to outputs \
   --nncf-quantization
```
Note that the number of epochs required for NNCF compression should not be set by command line parameter, since it is
calculated by the script `compress.py` itself.

The compressed model can be evaluated and exported to the OpenVINO™ format by the same commands as non-compressed model,
see the sections 6 and 7 above.
