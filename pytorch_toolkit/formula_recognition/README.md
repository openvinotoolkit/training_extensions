# PyTorch realization of the Im2Markup

This repository contains inference and training code for Im2LaTeX models.
Source [repository](https://github.com/harvardnlp/im2markup/). This repository is based on a [PyTorch realization](https://github.com/luopeixiang/im2latex/)
Models code is designed to enable ONNX\* export and inference on CPU\GPU via OpenVINO™.

## Setup

### Prerequisites

* Ubuntu\* 16.04
* Python\* 3.7 or newer
* PyTorch\* (1.4.0)
* CUDA\* 10.1
* OpenVINO™ 2020.1 with Python API

### Installation

Create and activate virtual environment:

```bash
bash init_venv.sh
```


### Download Datasets

For training model one has to have dataset. Dataset format is similiar to [im2latex-100k](https://zenodo.org/record/56198#.X2NDQ2gzaUl). Main structure of the dataset is following:
* `formulas_file` - file with one formula per line
* `images_folder` - folder containing input images
* `split_file` - this file contains `file_name` (tab symbol) `formula_idx` per line connecting corresponding index of the formula in the file with formulas and particular image with `image_name`. Example:
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

To train formula-recognition model run:

```bash
python3 tools/train.py --config configs/train_config.yml --work_dir <path to work dir>
```
Work dir is used to store information about learning: saved model checkpoints, logs.

### Description of possible options in train config:
 - `backbone_config`:
    * `arch`: type of the architecture (if backbone_type is resnet). For mor details, please, refer to [ResnetLikeBackBone](im2latex/models/backbones/resnet.py)
    * `disable_layer_3` and `disable_layer_4` - disables layer 3 and 4 in resnet-like backbone
    * `enable_last_conv` - enables additional convolution layer to adjust number of output channels to the number of input channels in the LSTM
    * `in_lstm_ch` - number of input LSTM channels, used for `last_conv`
- `backbone_type`: `resnet` for resnet-like backbone or anything else for original backbone from [im2markup](https://arxiv.org/pdf/1609.04938.pdf) paper
- `batch_size` - batch size used for training
- `device` - device for training, used in pytorch .to() method. Possible options: 'cuda', 'cpu', etc. `cpu` is used by default.
- `head` - configuration of the text recognition head
    * `beam_width` - witdth used in beam search. 0 - do not use beam search, 1 and more - use beam search with corresponding number of possible tracks.
    * `dec_rnn_h` - number of channels in decoding
    * `emb_size` - dimension of the embedding
    * `enc_rnn_h` - number of channels in encoding
    * `in_lstm_ch` - number of input in lstm channels, should be equal to `backbone_config.in_lstm_ch`
    * `max_len` - maximum possible length of the predicted formula
    * `n_layer` - describe
- `learning_rate` - learining rate
- `log_path` - path to store training logs
- `model_path` - path for model (if one wants to aftertune model)
- `optimizer` - Adam or SGD
- `save_dir` - dir to save checkpoints
- `train_paths` - list of paths from where get training data (if more than one path is specified, datasets are concatenated). If one wants to concatenate more than one instance of the desirable dataset, this dataset should be specified several times.
- `val_path` - path for validation data
- `vocab_path` - path where vocab file is stored
- `train_transforms_list` and
- `val_transforms_list` - here you can describe set of desirable transformations for train and validation datasets respectively
- `epochs` - number of epochs to train
- `clip_grad` - maximum possible value for gradient
- `old_model` - use this flag if you want to traing model trained in the previous versions of this framework.


One can point to pre-trained model [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/text_spotter/model_step_200000.pth) inside configuration file to start training from pre-trained weights. Change `configs/train_config.yml`:
```
...
model_path: <path_to_weights>
...
```

If the model was marked `old_model`, that means the model was trained in older version of this framework (concretely, model checkpoint keys are different from keys used in model now), so if you want to use this model in any context, point out this fact in `config`:
```
...
old_model: true
...
```


## Evaluation

`tools/test.py` script is designed for quality evaluation of im2latex models.

### PyTorch

Config file of the evaluation is similiar to train config:

```bash
python tools/test.py --config configs/eval_config.yml
```
Evaluation process is the following:
1. Run the model and get predictions
2. Render predictions from the first step into images of the formulas
3. Compare images.
The third step is very important because im LaTeX language one can write different formulas that are looking the same. Example:
`s^{12}_{i}` and `s_{i}^{12}` looking the same: both of them are rendered as ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20s%5E%7Bi%7D_%7B12%7D)
That is why we cannot just compare text predictions one-by-one, we have to render images and compare them.


## Demo

In order to see how trained model works using OpenVINO™ please refer to [Formula recognition Python* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/formula_recognition_demo/). Before running the demo you have to export trained model to IR. Please see below how to do that.

If you want to see how trained pytorch model is working, you can run `tools/demo.py` script with correct `config` file. Fill in the `input_images` variable with the paths to desired images. For every image in this list, model will predict the formula and print it into the terminal.

## Export PyTorch Models to OpenVINO™

To run the model via OpenVINO™ one has to export PyTorch model to ONNX first and
then convert to OpenVINO™ Intermediate Representation (IR) using Model Optimizer.

Model will be split into two parts:
- Encoder (cnn-backbone and part of the text recognition head)
- Text recognition decoder (LSTM + attention-based head)

### Export to ONNX*

The `tools/export.py` script exports a given model to ONNX representation.

```bash
python tools/export.py --config configs/export_config.yml
```


### Convert to IR

Conversion from ONNX model representation to OpenVINO™ IR is straightforward and
handled by OpenVINO™ Model Optimizer. Please refer to [Model Optimizer
documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) for details on how it works.

To convert model to IR one has to set flag `export_ir` in `config` file:
```
...
export_ir: true
...
```

If this flag is set, full pipeline (PyTorch -> onnx -> Openvino IR) is running, else model is exported to ONNX only.
