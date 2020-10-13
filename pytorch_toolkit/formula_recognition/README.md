# PyTorch realization of the Im2Markup

This repository is based on a [PyTorch realization](https://github.com/luopeixiang/im2latex/) of the code from the original [repository](https://github.com/harvardnlp/im2markup/).
Models code is designed to enable ONNX\* export and inference on CPU\GPU via OpenVINO™.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.7 or newer
* PyTorch\* (1.4.0)
* OpenVINO™ 2020.4 with Python API

### Installation

Create and activate virtual environment:

```bash
bash init_venv.sh
```

### Download Datasets

For training model one has to have dataset. Dataset format is similiar to [im2latex-100k](https://zenodo.org/record/56198#.X2NDQ2gzaUl). Main structure of the dataset is following:
* `formulas_file` - file with one formula per line
* `images_folder` - folder containing input images
* `split_file` - this file contains `image_name` (tab symbol) `formula_idx` per line connecting corresponding index of the formula in the file with formulas and particular image with `image_name`. Example:
    ```
    11.png  11
    34.png  34
    ```
    There should be at least two such files: `train_filter.lst` and `validate_filter.lst`

> **NOTE**:
> By default the following structure of the dataset is assumed:
> `images_processed` - folder with images
> `formulas.norm.lst` - file with preprocessed formulas, for details, refer to [im2markup](https://github.com/harvardnlp/im2markup) repository
> `validate_filter.lst` and `train_filter.lst` - corresponding splits of the data.


## Training

To train formula recognition model run:

```bash
python tools/train.py --config configs/config.yml --work_dir <path to work dir>
```
Work dir is used to store information about learning: saved model checkpoints, logs.

### Description of possible options in config:
The config file is divided into 4 sections: train, eval, export, demo. In every section there could exist common parameter as well as unique parameters.
#### Common parameters:
 - `backbone_config`:
    * `arch`: type of the architecture (if backbone_type is resnet). For more details, please, refer to [ResnetLikeBackBone](im2latex/models/backbones/resnet.py)
    * `disable_layer_3` and `disable_layer_4` - disables layer 3 and 4 in resnet-like backbone
    * `enable_last_conv` - enables additional convolution layer to adjust number of output channels to the number of input channels in the LSTM. Optional. Default: false.
    * `in_lstm_ch` - number of input LSTM channels, used only for `last_conv`.
- `backbone_type`: `resnet` for resnet-like backbone or anything else for original backbone from [im2markup](https://arxiv.org/pdf/1609.04938.pdf) paper. Optional. Default is `resnet`
- `head` - configuration of the text recognition head. All of the following parameters have default values, you can check them in [text reconition head](im2latex/models/text_recognition_heads/attention_based.py)
    * `beam_width` - witdth used in beam search. 0 - do not use beam search, 1 and more - use beam search with corresponding number of possible tracks.
    * `dec_rnn_h` - number of channels in decoding
    * `emb_size` - dimension of the embedding
    * `encoder_hidden_size ` - number of channels in encoding
    * `encoder_input_size ` - number of input in lstm channels, should be equal to `backbone_config.in_lstm_ch`
    * `max_len` - maximum possible length of the predicted formula
    * `n_layer` - describe
- `model_path` - path for model
- `val_path` - path for validation data
- `vocab_path` - path where vocab file is stored
- `val_transforms_list` - here you can describe set of desirable transformations for validation datasets respectively An example is given in the config file, for other options, please, refer to [constructor of transforms (section `create_list_of_transforms`)](im2latex/data/utils.py)
- `device` - device for training, used in PyTorch .to() method. Possible options: 'cuda', 'cpu', etc. `cpu` is used by default.
#### Training-specific parameters
In addition to common parameters you can specify the following arguments:
- `batch_size` - batch size used for training
- `learning_rate` - learining rate
- `log_path` - path to store training logs
- `optimizer` - Adam or SGD
- `save_dir` - dir to save checkpoints
- `train_paths` - list of paths from where get training data (if more than one path is specified, datasets are concatenated). If one wants to concatenate more than one instance of the desirable dataset, this dataset should be specified several times.
- `train_transforms_list` - similiar to `val_transforms_list`
- `epochs` - number of epochs to train

One can use some pretrained models. Right now two models are available:
* medium model:
    * [checkpoint link](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/medium_photograped_0185.pth)
    * digits, letters, some greek letters, fractions, trigonometric operations are supported; for more details, please, look at corresponding vocab file
    * to use this model, please, change these fields in the config file:
    ```
    backbone_config:
        arch: resnext50_32x4d
        disable_layer_3: true
        disable_layer_4: true
    model_path: <path to the model>
    vocab_path: vocabs/vocab_medium.json
    ```
The model can be used either for recognizing rendered formulas and for recognizing scanned formulas (e.g. from a scanner or from a phone camera)

* handwritten polynomials model:
    * [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/polynomials_handwritten_0166.pth)
    * digits, letters, upper indices are supported
    * to use this model, please, change these field in the config file:
    ```
    backbone_config:
        arch: resnext50_32x4d
        disable_layer_3: false
        disable_layer_4: true
    model_path: <path to the model>
    vocab_path: vocabs/vocab_handwritten_polynomials.json
    ```
The model can be used for recognizing handwritten polynomial equations.
All the above models can be used for aftertuning or as ready for inference models. To provide maximum quality at recognizing formulas, it is highly recommended to preprocess image - simply binarize it, you can find corresponding prepocessing at [this file](im2latex/data/utils.py). Just state the desired preprocessing at the corresponding section of the config file (train, eval, etc).

#### Evaluation-specific parameters
- `split_file` - name of the file with labels (note: physical file name should end with `_filter.lst`). Default is `validate`
- `target_metric` - target value of the metric. Used in tests. For test to pass, result value should be greater or equal than `target_metric`

#### Demo-specific parameters
- `input_images` - list of paths for input images

#### Export-specific parameters
These are parameters used for model export to ONNX & OpenVINO™ IR:
- `res_encoder_name` - filename to save the converted encoder model (with `.onnx` postfix)
- `res_decoder_name` - filename to save the converted decoder model (with `.onnx` postfix)
- `dummy_input` - input image used for tracing PT model and creating onnx graph. This image is also used to compare that outputs from ONNX model and PyTorch model are the same.
- `input_shape_decoder` - list of dimensions describing input shape for encoder for OpenVINO IR conversion.
- `export_ir` - Set this flag to `true` to export model to the OpenVINO IR. For details refer to [convert to IR section](#convert-to-ir)
- `verbose_export` - Set this flag to `true` to perform verbose export (i.e. print model optimizer commands to terminal)



## Evaluation

`tools/test.py` script is designed for quality evaluation of im2latex models.

### PyTorch

Config file of the evaluation is similiar to train config:

```bash
python tools/test.py --config configs/config.yml
```
Evaluation process is the following:
1. Run the model and get predictions
2. Render predictions from the first step into images of the formulas
3. Compare images.
The third step is very important because in LaTeX language one can write different formulas that are looking the same. Example:
`s^{12}_{i}` and `s_{i}^{12}` looking the same: both of them are rendered as ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20s%5E%7Bi%7D_%7B12%7D)
That is why we cannot just compare text predictions one-by-one, we have to render images and compare them.


## Demo

In order to see how trained model works using OpenVINO™ please refer to [Formula recognition Python* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/formula_recognition_demo/). Before running the demo you have to export trained model to IR. Please see below how to do that.

If you want to see how trained PyTorch model is working, you can run `tools/demo.py` script with correct `config` file. Fill in the `input_images` variable with the paths to desired images. For every image in this list, model will predict the formula and print it into the terminal.

## Export PyTorch Models to OpenVINO™

To run the model via OpenVINO™ one has to export PyTorch model to ONNX first and
then convert to OpenVINO™ Intermediate Representation (IR) using Model Optimizer.

Model will be split into two parts:
- Encoder (CNN-backbone and part of the text recognition head)
- Text recognition decoder (LSTM + attention-based head)

### Export to ONNX*

The `tools/export.py` script exports a given model to ONNX representation.

```bash
python tools/export.py --config configs/config.yml
```


### Convert to IR

Conversion from ONNX model representation to OpenVINO™ IR is straightforward and handled by OpenVINO™ Model Optimizer.

To convert model to IR one has to set flag `export_ir` in `config` file:
```
...
export_ir: true
...
```

If this flag is set, full pipeline (PyTorch -> ONNX -> Openvino™ IR) is running, else model is exported to ONNX only.
