# Noise Suppression

This is an implementation of the train script in PyTorch\*.
The script allows to train a noise suppression model on the [DNS-Challenge dataset](https://github.com/microsoft/DNS-Challenge).
The model architecture was inspired by [PoCoNet](https://arxiv.org/abs/2008.04470) model


## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train model](#train-model)
4. [Convert a Model to OpenVINO™ format for Demo](#convert-a-models-to-openvino-format-for-demo)


## Requirements

To run the script Python\* 3.6 has to be installed on the machine.
To install the required packages, run the following:

```bash
pip install -r requirements.txt
```

## Preparation

The [DNS-Challenge dataset](https://github.com/microsoft/DNS-Challenge) is used to train the model.
You can get the data by cloning the repository
```git clone https://github.com/microsoft/DNS-Challenge.git <dns-challenge-dir>```

The train script also allows to evaluate model on synthetic ICASSP_dev_test_set. The 'icassp2021-final' branch of the DNS-Challenge repo has to be checked-out to enable this option


## Train Model

After you get the data, you can train the model.
To do this you may use the following command line:

```bash
python3 train.py \
--model_desc=config.json \
--output_dir=result_model \
--num_train_epochs=20 \
--size_to_read=4 \
--per_gpu_train_batch_size=6 \
--total_train_batch_size=128 \
--dns_datasets=<dns-challenge-dir>/datasets \
--eval_data=<dns-challenge-dir>/datasets/ICASSP_dev_test_set/track_1/synthetic
```

As a result the trained model has to be located in newly created folder 'result_model'.

## Convert a Models to OpenVINO™ format for Demo

The script also saves the ONNX\* model into the same output folder together with the PyTorch\* model . This ONNX\* model can be converted to OpenVINO™ format

```bash
mo.py --input_model result_model/model.onnx
```

After conversion to the OpenVINO™ format you can try model using
the [demo for noise suppresion models](https://github.com/openvinotoolkit/open_model_zoo/tree/develop/demos/noise_suppression_demo/python)
