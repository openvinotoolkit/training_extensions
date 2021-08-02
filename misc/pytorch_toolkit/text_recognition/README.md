# PyTorch Text Recognition

This is PyTorch implementation of some text recognition models.

This code is based on this [repo](https://github.com/luopeixiang/im2latex/).

Models code is designed to enable ONNX\* export and inference on CPU\GPU via OpenVINO™.

## Supported Tasks
Two tasks are supported:
1. LaTeX formula recognition.
2. Alphanumeric scene text recognition.

## Model Architecture

We follow similar to [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) approach of defining model architecture, but we do not divide the last two steps. Thus, every model consists of three steps:
1. Image rectification. We use Thin-Plate-Spline from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
2. Convolutional backbone:
   1. ResNet-like backbone
   2. Custom ResNet-like backbone (configurable number of channels and spatial dimension in every stage)
    > [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) is implemented for the resnet-like backbone, but currently it is not used by any of the models.
3. Text Recognition Head
   1. CTC lstm encoder-decoder head.
   2. 1d attention based head from [im2markup](https://arxiv.org/pdf/1609.04938.pdf)
   3. 2d attention based head from [YAMTS](https://arxiv.org/abs/2106.12326)

## Setup

### Prerequisites

* Ubuntu\* 18.04
* Python\* 3.7 or newer
* PyTorch\* (1.5.1)
* OpenVINO™ 2021.4 with Python API

### Optional prerequisites

#### Install required packages for evaluation (only for formula-recognition)
These packages are used for rendering images while evaluation and demo.

```bash
sudo apt-get update &&
  sudo apt-get install -y --no-install-recommends \
    texlive \
    imagemagick \
    ghostscript
```

<details>
 <summary> Known issue with imagemagick  </summary>
Evaluation process uses imagemagick to convert PDF-rendered formulas into PNG images. Sometimes there could be errors:

```
convert-im6.q16: not authorized `/tmp/tmpgr1m4d4_.pdf' @ error/constitute.c/ReadImage/412.
convert-im6.q16: no images defined `/tmp/tmpgr1m4d4_.png' @ error/convert.c/ConvertImageCommand/3258.
```
The problem is missing required permissions.
To fix this open file `/etc/ImageMagick-6/policy.xml`:

`sudo nano /etc/ImageMagick-6/policy.xml`

Find `<policy domain="coder" rights="none" pattern="PDF" />`

and replace with:

`<policy domain="coder" rights="read|write" pattern="PDF" />`

</details>

### Installation

Create and activate virtual environment:

```bash
bash init_venv.sh
```

### Download or Prepare Datasets

#### Dataset Format

Several dataset formats are supported:

1. LMDB dataset.
   Dataset in form of LMDB database is supported. This is the preferred dataset format (for alphanumeric scene text recognition). You can download prepared data from [deep text recognition benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
   Example of usage:
   ```yaml
   type: LMDBDataset
   case_sensitive: false
   data_path: evaluation/IC13_1015
   grayscale: true
   fixed_img_shape:
     - 32
     - 120
   ```
2. Im2latex format.
   This dataset is used to train formula recognition models.
   Dataset format is similar to [im2latex-100k](https://zenodo.org/record/56198#.X2NDQ2gzaUl). Main structure of the dataset is following:
   * `formulas.norm.lst` - file with one formula per line.
   * `imaged_processed` - folder containing input images.
   * `split_file` - this file contains `image_name` (tab symbol) `formula_idx` per line connecting corresponding index of the formula in the file with formulas and particular image with `image_name`. Example:
       ```
       11.png  11
       34.png  34
       ...
       ```
       There should be at least two such files: `train_filter.lst` and `validate_filter.lst`

    You can prepare your own dataset in the same format as above.
    Samples of the dataset can be found [here](../../../data/formula_recognition).

    > **NOTE**:
    > By default the following structure of the dataset is assumed:
    > `images_processed` - folder with images
    > `formulas.norm.lst` - file with preprocessed formulas. If you want to use your own dataset, formulas should be preprocessed. For details, refer to [this script](https://github.com/harvardnlp/im2markup/blob/master/scripts/preprocessing/preprocess_formulas.py).
    > `validate_filter.lst` and `train_filter.lst` - corresponding splits of the data.
3. ICDAR13 recognition dataset.
   See details [here](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads)

4. CocoLike dataset.
   СocoLike annotation is supported. See [here](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/datasets/open_images_v5_text) for details.
    Example of usage:
   ```yaml
   type: CocoLikeDataset
   case_sensitive: false
   data_path: OpenImagesV5
   annotation_file: train_1.json
   grayscale: true
   fixed_img_shape:
     - 32
     - 120
   ```


Every dataset class has its own constructor with specific parameters. You can see **constructors** [here](./text_recognition/datasets/dataset.py). Examples of use of different datasets can be seen in the config files:
* [example](./configs/config_0014.yml)
* [example](./configs/medium_config.yml)


#### Vocabulary files

When you prepare your own dataset with `formulas.norm.lst` file, you will have to create a vocabulary file for this dataset.
Vocabulary file is a special file which is used to cast token ids to human readable tokens and vice versa.
Like letters and digits in the natural language, tokens here are atomic units of the latex language (e.g. `\\sin`, `1`, `\\sqrt`, etc).
You can find an example in the [vocabs folder](./vocabs/) of this project.
Use [this script](./tools/make_vocab.py) to create vocab file from your own formulas file.
The script will read the formulas and create the vocabulary from the formulas used in train split of the dataset.
> If you use one of the general text recognition datasets (such as ICDAR13 or synth90k), vocab file is already prepared. You can find it in the `vocabs` folder.



## Training

To train text recognition model run:

```bash
python tools/train.py --config <path to config> --work_dir <path to work dir>
```
Work dir is used to store information about learning: saved model checkpoints, logs.

## Evaluation

`tools/test.py` script is designed for quality evaluation of the text-recognition models.

### Model Zoo
Here you can find which models are currently supported. For accuracy metrics and converted to OpenVINO IR model complexity info, please, refer to OMZ page of the specific model.
| Model Name                   | Task                                | Transformation | Backbone         | Head                     | Link to OMZ                                                                                                                         | Checkpoint link                                                                                                                         | Config File                                        |
| ---------------------------- | ----------------------------------- | -------------- | ---------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| medium-rendered-0001         | Formula Recognition                 | None           | ResNeXt-50 st 2  | 1d attention             | [link](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/formula-recognition-medium-scan-0001)             | [link](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/medium_photograped_0185.pth)              | [link](configs/medium_config.yml)                  |
| polynomials-handwritten-0001 | Formula Recognition                 | None           | ResNeXt-50 st 3  | 1d attention             | [link](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/formula-recognition-polynomials-handwritten-0001) | [link](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/polynomials_handwritten_0166.pth)         | [link](configs/polynomials_handwritten_config.yml) |
| text-recognition-0014        | Alphanumeric Scene Text Recognition | None           | ResNeXt-50 st 2  | CTC LSTM Encoder-Decoder | [link](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/text-recognition-0014)                            | [link](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/text_recognition/text_recognition_0014.pth) | [link](configs/config_0014.yml)                    |
| text-recognition-0015        | Alphanumeric Scene Text Recognition | None           | ResNeXt-101      | 2d attention             | [link](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/text-recognition-0015)                            | [link](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/text_recognition/text_recognition_0015.pth) | [link](configs/config_0015.yml)                    |
| text-recognition-0016 ([YATR](https://arxiv.org/abs/2107.13938)) | Alphanumeric Scene Text Recognition | TPS            | ResNeXt-101 st 3 | 2d attention             | tbd                                                                                                                                 | [link](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/text_recognition/text_recognition_0016.pth) | [link](configs/config_0016.yml)                    |

> Note: st* for ResNeXt models stands for stage. This means that only first several stages of the ResNeXt like backbone are used.
### PyTorch

For example, one can run evaluation process using config for `medium` model.
```bash
python tools/test.py --config configs/medium_config.yml
```
Evaluation process is the following:
1. Run the model and get predictions
1. (optionally) Render predictions from the first step into images of the formulas
2. Compare images if `render` flag is true, else just compare predicted and GT text.
> The third step is important for LaTeX models because in LaTeX language one can write different formulas that are looking the same. Example:
`s^{12}_{i}` and `s_{i}^{12}` looking the same: both of them are rendered as ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20s%5E%7Bi%7D_%7B12%7D)
That is why we cannot just compare text predictions one-by-one, we have to render images and compare them.


## Demo

In order to see how trained model works using OpenVINO™ please refer to [Formula recognition Python\* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/formula_recognition_demo/) and [Text detection C++\* demo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/text_detection_demo). Before running the demo you have to export trained model to IR. Please, see below how to do that.

If you want to see how trained PyTorch model is working, you can run `tools/demo.py` script with correct `config` file. Fill in the `input_images` variable with the paths to desired images. For every image in this list, model will predict the text and print it into the terminal.

## Export PyTorch Models to OpenVINO™

To run the model via OpenVINO™ one has to export PyTorch model to ONNX first and
then convert to OpenVINO™ Intermediate Representation (IR) using Model Optimizer.

Model will be split into two parts if it has Attention head:
- Encoder (CNN-backbone and part of the text recognition head)
- Text recognition decoder

Else the model will be exported as one file.

### Export to ONNX*

The `tools/export.py` script exports a given model to ONNX representation.

```bash
python tools/export.py --config configs/medium_config.yml
```


### Convert to IR

Conversion from ONNX model representation to OpenVINO™ IR is straightforward and handled by OpenVINO™ Model Optimizer.

To convert model to IR one has to set flag `export_ir` in `config` file:
```
...
export_ir: true
...
```

If this flag is set, full pipeline (PyTorch -> ONNX -> Openvino™ IR) is running, else model is exported to the ONNX\* only.

### Tests

There are 3 groups of tests for every supported configuration:
* Train test
* Evaluation test
* Export test

To run tests:
```bash
# cd to the dir where this README.md is placed (text_recognition)
# if you are in the root of the training_extensions repo:
cd misc/pytorch_toolkit/text_recognition
# activate venv and run tests:
pytest tests

```
