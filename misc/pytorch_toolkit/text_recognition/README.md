# PyTorch Text Recognition

This folder contains code related to PyTorch text recognition.
There are three subtasks supported by this code:
* general alphanumeric text recognition
* two specific formula-recognition tasks:
  * recognition of the handwritten polynomial equations
  * recognition of the rendered and scanned printed formulas
    This code is based on a [PyTorch realization](https://github.com/luopeixiang/im2latex/) of the code from the original [repository](https://github.com/harvardnlp/im2markup/).

Models code is designed to enable ONNX\* export and inference on CPU\GPU via OpenVINO™.

## Setup

### Prerequisites

* Ubuntu\* 18.04
* Python\* 3.7 or newer
* PyTorch\* (1.5.1)
* OpenVINO™ 2021.2 with Python API

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

#### Known issue with imagemagick
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

### Installation

Create and activate virtual environment:

```bash
bash init_venv.sh
```

### Download or Prepare Datasets

#### Dataset Format

Several dataset formats are supported:

1. Im2latex format.
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
    Samples of the dataset can be found [here](../../data/formula_recognition).

    > **NOTE**:
    > By default the following structure of the dataset is assumed:
    > `images_processed` - folder with images
    > `formulas.norm.lst` - file with preprocessed formulas. If you want to use your own dataset, formulas should be preprocessed. For details, refer to [this script](https://github.com/harvardnlp/im2markup/blob/master/scripts/preprocessing/preprocess_formulas.py).
    > `validate_filter.lst` and `train_filter.lst` - corresponding splits of the data.
2. ICDAR13 recognition dataset.
   See details [here](http://dagdata.cvc.uab.es/icdar2013competition/?ch=2&com=downloads)

3. Synth90k dataset (MJSynth)
   See details [here](https://www.robots.ox.ac.uk/~vgg/data/text/)

4. IIIT5k
   See details [here](https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)

Every dataset class has its own constructor with specific parameters. You can see costructors [here](./text_recognition/datasets/dataset.py). Examples of use of different datasets can be seen in the config files:
* [example](./configs/config_0013.yml)
* [example](./configs/medium_config.yml)


#### Vocabulary files

When you prepare your own dataset with `formulas.norm.lst` file, you will have to create a vocabulary file for this dataset.
Vocabulary file is a special file which is used to cast token ids to human readable tokens and vice versa.
Like letters and digits in the natural language, tokens here are atomic units of the latex language (e.g. `\\sin`, `1`, `\\sqrt`, etc).
You can find an example in the [vocabs folder](./vocabs/) of this project.
Use [this script](./tools/make_vocab.py) to create vocab file from your own formulas file.
The script will read the formulas and create the vocabulary from the formulas used in train split of the dataset.
> If you use one of the general text recognition datasets (such as ICDAR13 or synth90k), vocab file is already prepared. You can find it [here](./vocabs/vocab_alphanumeric_ctc.json)



## Training

To train text recognition model run:

```bash
python tools/train.py --config <path to config> --work_dir <path to work dir>
```
Work dir is used to store information about learning: saved model checkpoints, logs.

### Description of possible options in config:
The config file is divided into 4 sections: train, eval, export, demo. Common parameters (like path to the model) are stored on the same level as train and other sections. Unique parameters (like learning rate) are stored in specific sections. Unique parameters and common parameters are mutually exclusive.
> **Note**: All values in the config file which have 'path' in their name will be treated as paths and the script which reads configuration will try to resolve all relative paths. By default all relative paths are resolved relatively to the folder where this README.md file is placed. Keep this in mind or use full paths.
#### Common parameters:
- `backbone_config`:
    * `arch`: type of the architecture (if backbone_type is resnet). For more details, please, refer to [ResnetLikeBackBone](im2latex/models/backbones/resnet.py)
    * `disable_layer_3` and `disable_layer_4` - disables layer 3 and 4 in resnet-like backbone. ResNet backbone from the torchvision module consists of 4 block of layers, each of them increase the number of channels and decrease the spatial dimensionality. These parameters allow to switch off the 3rd and the 4th of such layers, respectively.
    * `enable_last_conv` - enables additional convolution layer to adjust number of output channels to the number of input channels in the LSTM. Optional. Default: false.
    * `output_channels` - number of output channels channels. If `enable_last_conv` is `true`, this parameter should be equal to `head.encoder_input_size`, otherwise it should be equal to actual number of output channels of the backbone.
- `backbone_type`: `resnet` for resnet-like backbone or anything else for original backbone from [im2markup](https://arxiv.org/pdf/1609.04938.pdf) paper. Optional. Default is `resnet`
- `head` - configuration of the text recognition head. All of the following parameters have default values, you can check them in [text reconition head](im2latex/models/text_recognition_heads/attention_based.py)
    * `beam_width` - width used in beam search. 0 - do not use beam search, 1 and more - use beam search with corresponding number of possible tracks.
    * `dec_rnn_h` - number of channels in decoding
    * `emb_size` - dimension of the embedding
    * `encoder_hidden_size ` - number of channels in encoding
    * `encoder_input_size ` - number of channels in the lstm input, should be equal to `backbone_config.output_channels`
    * `max_len` - maximum possible length of the predicted formula
    * `n_layer` - number of layers in the trainable initial hidden state for each row
- `model_path` - path to the pretrained model checkpoint (you can find the links to the checkpoints below in this document).
- `vocab_path` - path where vocabulary file is stored.
- `val_transforms_list` - here you can describe set of desirable transformations for validation datasets respectively. An example is given in the config file, for other options, please, refer to [constructor of transforms (section `create_list_of_transforms`)](im2latex/data/utils.py)
- `device` - device for training, used in PyTorch .to() method. Possible options: 'cuda', 'cpu'. `cpu` is used by default.
#### Training-specific parameters
In addition to common parameters you can specify the following arguments:
- `batch_size` - batch size used for training
- `learning_rate` - learining rate
- `log_path` - path to store training logs
- `optimizer` - Adam or SGD
- `save_dir` - dir to save checkpoints
- `datasets` - list of datasets which will be used in training. Common parameters are:
  - `type` - name of the dataset, for details see [here](./text_recognition/datasets/dataset.py). Dataset names are described in the `str_to_class` section
  - `subset` - how to use this subset. Options: `train` or `validate`
    Any other parameters are dataset specific and should be set in correspondance with its constructor.
- `train_transforms_list` - similar to `val_transforms_list`
- `epochs` - number of epochs to train

One can use some pretrained models. Right now three models are available:
* medium model:
    * [checkpoint link](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/medium_photograped_0185.pth)
    * digits, letters, some greek letters, fractions, trigonometric operations are supported; for more details, please, look at [corresponding vocab file](vocabs/vocab_medium.json).
    * to use this model, just set the correct value to the `model_path` field in the bcorresponding config file:
    ```
    model_path: <path to the model>
    ```
The model can be used for recognizing both rendered and scanned formulas (e.g. from a scanner or from a phone camera)

* handwritten polynomials model:
    * [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/formula_recognition/polynomials_handwritten_0166.pth)
    * digits, letters, upper indices are supported
    * to use this model, please, change model path in the corresponding config file:
    ```
    model_path: <path to the model>
    ```
The model can be used for recognizing handwritten polynomial equations.

* alphanumeric model
  * [checkpoint]
  * number from 0 to 9 and latin letters in lower case are supported
   * to use this model, please, change model path in the corresponding config file:
    ```
    model_path: <path to the model>
    ```

All the above models can be used for aftertuning or as ready for inference models. To provide maximum quality at recognizing text, it is highly recommended to preprocess image - simply binarize it:
```
val_transform_list:
    - name: TransformBin
      threshold: 100
```
You can find other prepocessing at [this file](im2latex/data/utils.py).
Some of sample images in the [data](../../data) section of this repo are already preprocessed, you can look at the examples.


#### Evaluation-specific parameters
- `dataset` - the same as in `train` section, but here it is just one dataset, so it does not have `subset` section.
- `render` - render images to compare them or just compare predicted and ground-truth text. By default is `true`. Used only for formula recognition. See Evaluation section for details.

#### Demo-specific parameters
- `transforms_list` - list of image transformations (optional)

#### Export-specific parameters
These parameters are used for model export to ONNX & OpenVINO™ IR:
- In case model is divided into encoder and decoder (for models with attention head, like formula recognition models):
  - `res_encoder_name` - filename to save the converted encoder model (with `.onnx` postfix)
  - `res_decoder_name` - filename to save the converted decoder model (with `.onnx` postfix)
- Else if model is monolithic (for models with CTC-head, like alphanumeric recognition):
  - `res_model_name` - filename to save the converted model (with `.onnx` postfix)

- `export_ir` - Set this flag to `true` to export model to the OpenVINO IR. For details refer to [convert to IR section](#convert-to-ir)
- `verbose_export` - Set this flag to `true` to perform verbose export (i.e. print model optimizer commands to terminal)
- `input_shape_decoder` for composite (encoder-decoder) or `input_shape` for monolithic model  - list of dimensions describing input shape for encoder for OpenVINO IR conversion.



## Evaluation

`tools/test.py` script is designed for quality evaluation of formula-recognition models.

### PyTorch

For example, one can run evaluation process using config for `medium` model.
```bash
python tools/test.py --config configs/medium_config.yml
```
Evaluation process is the following:
1. Run the model and get predictions
1. (optionally) Render predictions from the first step into images of the formulas
2. Compare images if `render` flag is true, else just compare predicted and GT text.
> The third step is very important because in LaTeX language one can write different formulas that are looking the same. Example:
`s^{12}_{i}` and `s_{i}^{12}` looking the same: both of them are rendered as ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20s%5E%7Bi%7D_%7B12%7D)
That is why we cannot just compare text predictions one-by-one, we have to render images and compare them.


## Demo

In order to see how trained model works using OpenVINO™ please refer to [Formula recognition Python\* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/formula_recognition_demo/) and [Text detection C++\* demo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/text_detection_demo). Before running the demo you have to export trained model to IR. Please, see below how to do that.

If you want to see how trained PyTorch model is working, you can run `tools/demo.py` script with correct `config` file. Fill in the `input_images` variable with the paths to desired images. For every image in this list, model will predict the formula and print it into the terminal.

## Export PyTorch Models to OpenVINO™

To run the model via OpenVINO™ one has to export PyTorch model to ONNX first and
then convert to OpenVINO™ Intermediate Representation (IR) using Model Optimizer.

Model will be split into two parts if it has Attention head:
- Encoder (CNN-backbone and part of the text recognition head)
- Text recognition decoder (LSTM + attention-based head)

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

If this flag is set, full pipeline (PyTorch -> ONNX -> Openvino™ IR) is running, else model is exported to ONNX only.
