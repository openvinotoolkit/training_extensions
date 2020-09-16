# PyTorch realization of the Im2Markup

This repository contains inference and training code for Im2LaTeX models.
Source [repository](https://github.com/harvardnlp/im2markup/). This repository is a fork of [PyTorch realization](https://github.com/luopeixiang/im2latex/)
Models code is designed to enable ONNX\* export and inference on CPU via OpenVINO™.

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

Links to our datasets????


## Training

To train Text Spotter model run:

```bash
python3 tools/train.py --config configs/train_config.yml
```

One can point to pre-trained model [checkpoint](https://download.01.org/opencv/openvino_training_extensions/models/text_spotter/model_step_200000.pth) inside configuration file to start training from pre-trained weights. Change `configs/train_config.yml`:
```
...
model_path: <path_to_weights>
...
```

If the model was marked `old_model`, that means that means that model was trained in older version of this framework (concretly, model checkpoint keys are different from keys used in model now), so if you want to use this model in any context, point out this fact in `config`:
```
...
old_model: true
...
```


## Evaluation

`tools/test.py` script is designed for quality evaluation of im2latex models.

### PyTorch

For example, to evaluate text-spotting-0001 model on medium_v2 test dataset
using PyTorch backend run:

```bash
python tools/test.py --config configs/eval_config.yml
```


## Demo

In order to see how trained model works using OpenVINO™ please refer to [Formula recognition Python* Demo](https://github.com/opencv/open_model_zoo/tree/develop/demos/python_demos/formula_recognition_demo/). Before running the demo you have to export trained model to IR. Please see below how to do that.

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