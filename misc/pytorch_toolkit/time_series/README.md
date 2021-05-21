# Time Series Forecasting

This is implementation of the Time Series Forecasting training framework in PyTorch\*.
The repository contains a time series forecasting method based on Temporal Fusion Transformer compatible with OpenVINO for fast inference.

## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)

## Requirements

The code is tested on Python\* 3.6.9, with dependencies listed in the `requirements.txt` file.
To install the required packages, run the following:

```bash
pip install -r requirements.txt
```

## Preparation

Currently we support Electricity dataset only.
To train a forecasting model with your dataset, you should implement a Dataset class containing data-specific preparation following next steps:

1. Create new dataset class in core/dataset/
``` python
from torch.utils.data import Dataset

class YourOwnDataset(Dataset):
    @staticmethod
    def get_split(*args, **kwargs):
        """ Returns train, val, test datasets of YourOwnDataset type.
        """
        ...
        return train, val, split

    def __len__(self):
        """ Returns size of current dataset
        """
        return len_of_your_dataset

    def __getitem__(self, idx):
        """ Returns sample from dataset.

        Returns:
        inputs (np.ndarray with shape [input_timestamps, n_input_features]): input data
        outputs: (np.ndarray with shape [output_timestamps, n_output_features]): output data
        mean: (float): normalization coefficients of sample (0 if you don't use any normalizations).
        scale: (float): normalization coefficients of sample (1 if you don't use any normalizations).
        """
        ...
        return inputs, outputs, mean, scale
```

2. Register your class in core/datasets/__init__.py
``` python

from .electricity_dataset import *
from .your_own_dataset import *

DATASETS = {
    "your_own_dataset": YourOwnDataset
}

def get_dataset(dataset_name):
    assert dataset_name in DATASETS
    return DATASETS[dataset_name]

```

3. Now you can train forecasting model with your own dataset.


## Train/Eval

After you prepared the data, you can train or validate your model.
Use commands below as an example.

### Prepare config file

To train your model, you should prepare a config file in .json format (for example we use parameters for Electricity dataset):

```json
{
    "dataset": {
        "name": "electricity",
        "params": {
            "data_folder": "<path_to_data_folder>"
        }
    },
    "pipeline": {
        "batch_size": 64,
        "num_workers": 8,
        "lr": 0.001,
        "milestones": [50],
        "epochs": 100
    },
    "model": {
        "num_categorical_variables": 1, # number of categorical features
        "num_regular_variables": 4, # number of regular features
        "category_counts": [369], # number of categories of the each categorical feature
        "hidden_size": 160, # size of hidden state
        "dropout": 0.1, # dropout ratio
        "num_heads": 4, # number of self-attention heads
        "output_size": 1, # number of predicted features for each output timestamp
        "quantiles": [0.1, 0.5, 0.9], # quantiles that will be predicted
        "num_encoder_steps": 168, # number of input timestamps
        "input_obs_idx": [0], # indexes of observed features
        "static_input_idx": [4],  # indexes of static input features
        "known_categorical_input_idx": [0], # indexes of categorical input features
        "known_regular_input_idx": [1, 2, 3] # indexes of regular input features
    }
}
```

### Train a Model

```bash
python3 train.py --gpus <n_gpus> --cfg <path_to_json_config> --val-chcek-interval <evaluate_model_each_n_iterations> --output-dir <folder_to_store_ckpts> --log-path <path_to_store_json_log_file>
```

### Start training from checkpoint

```bash
python3 train.py --gpus <n_gpus> --cfg <path_to_json_config> --val-chcek-interval <evaluate_model_each_n_iterations> --output-dir <folder_to_store_ckpts> --log-path <path_to_store_json_log_file> --ckpt <path_to_ckpt>
```

### Evaluate a Model

Our framework supports automatic best model selection based on Normalized Quantile Loss score evaluation in training time.
Evaluate your model without training using below command:

```bash
python3 train.py --gpus <n_gpus> --cfg <path_to_json_config> --val-chcek-interval <evaluate_model_each_n_iterations> --output-dir <folder_to_store_ckpts> --log-path <path_to_store_json_log_file> --ckpt <path_to_ckpt> --test
```

#### Convert a Model to the ONNX\* and OpenVINO™ format

PyTorch to ONNX:
```bash
python3 train.py --cfg <path_to_json_config> --ckpt <path_to_ckpt> --to-onnx <path_to_output_onnx>
```

ONNX to OpenVINO™:
```bash
<path_to_model_optimizer_folder>/mo.py --input_model <path_to_output_onnx>'
```

## Demo

You can try your models after converting them to the OpenVINO™ format or a [pretrained model from OpenVINO™](https://docs.openvinotoolkit.org/latest/machine_translation.html) using the [demo application from OpenVINO™ toolkit](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_machine_translation_demo_README.html)