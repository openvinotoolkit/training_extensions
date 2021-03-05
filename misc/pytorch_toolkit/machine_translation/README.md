# Machine Translation

This is implementation of the Neural Machine Translation training framework in PyTorch\*.
The repository contains a non-autoregressive neural machine translation model LaNMT compatible with OpenVINO for fast translation.

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

Preprocess train/val dataset from your own bilingual corpus as described in the section below.

There are two ways to train the model for special lingual pair:
1. Using existed bilingual corpus.
2. Using pseudo labeling techniques for monolingual corpus.

### Notation

* A (monolingual) corpus is the collection of sentences for some language.
* A bilingual corpus is the collection of sentences for two languages aligned with each other.


### TXT -> LMDB conversion

There are two formats of the corpus representation that supports in our framework:
1. Plain text format is the most simple format of the text representation.
2. LMDB is the more faster format than plain text with less requirements for RAM. LMDB is recommended to use with large corpuses.

Convert corpus files from plain text to LMDB format using the provided Python script:

```bash
python3 txt_to_lmdb.py --names <path_to_corpus_0>,<path_to_corpus_1>,.., <path_to_corpus_n> --output-lmdb <path_to_store_lmdb>
```

Convert corpus files from LMDB to plain text format using the provided Python script:

```bash
python3 lmdb_to_txt.py --input <path_to_input_lmdb> --output <path_to_store_output_txt>
```


### Distillation

If you have an existed bilingual corpus, you can skip this step.
But we should notice that distillation is a powerful technique for train machine translation models.
There are two common cases where you can use distillation:
1. There are many works where noticed that you can achieve the best score using distillation from source to target language instead of real target corpus.
2. Distillation can be used for augmentation of training dataset using the backtranslation technique when you translate from target to source corpus.
3. If you have a monolingual corpus only, you can create a bilingual corpus using existed translation models.

There is the script `distiller.py` to make distillation, which using MarianMT models as a teacher.
The list of supported language pairs can be found [here](https://huggingface.co/Helsinki-NLP).
You can use TXT or LMDB format as input, but there is only one LMDB format available as output.

Distillation corpus from source to target language using the provided Python script:

```bash
python3 distiller.py --langs <src>-<tgt> --gpus <n_gpus> --batch-size <batch_size> --input-corpus <path_to_input_corpus> --input-format <txt|lmdb> --output-lmdb <path_to_store_lmdb>
```

### Train tokenizer

To train your model, you should prepare a tokenizer using your corpus using the provided Python script:

```bash
python3 train_tokznier.py --coprus-list <path_to_corpus_list> --vocab-size <vocab_size> --output-folder <path_to_output>
```

--corpus-list - path to txt file that contains list of corpus files that that will be used for training of the tokenizer.
There is only txt format supports to train tokenizer.

## Train/Eval

After you prepared the data, you can train or validate your model.
Use commands below as an example.

### Prepare config file

To train your model, you should prepare a config file in .json format:

```json
{
    "tokenizer": {
        "src": {
            "type": "spbpe",
            "params": {
                "path": "<path_to_src_tokenizer_folder>",
                "max_length": 150,
                "enable_truncation": true,
                "enable_padding": true
            }
        },
        "tgt": {
            "type": "spbpe",
            "params": {
                "path": "<path_to_tgt_tokenizer_folder>",
                "max_length": 150,
                "enable_truncation": true,
                "enable_padding": true
            }
        }
    },
    "model": {
        "type": "lanmt",
        "params": {
            "embed_size": 512,
            "hidden_size": 512,
            "n_att_heads": 8,
            "prior_layers": 6,
            "latent_dim": 8,
            "q_layers": 6,
            "max_delta": 50,
            "decoder_layers": 6,
            "kl_budget": 1.0,
            "budget_annealing": false,
            "max_steps": 1,

        }
    },
    "trainset": {
        "src": {
            "type": "txt",
            "corpus": "<path_to_train_src_corpus>"
        },
        "tgt": {
            "type": "txt",
            "corpus": "<path_to_train_tgt_corpus>"
        }
    },
    "valset": {
        "src": {
            "type": "txt",
            "corpus": "<path_to_val_src_corpus>"
        },
        "tgt": {
            "type": "txt",
            "corpus": "<path_to_val_tgt_corpus>"
        }
    },
    "trainer": {
        "lr": 0.0003,
        "milestones": "20,40",
        "batch_size": 15,
        "num_workers": 8,
        "max_epochs": 80
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

Our framework supports automatic best model selection based on BLEU score evaluation in training time.
Evaluate your model without training using below command:

```bash
python3 train.py --gpus <n_gpus> --cfg <path_to_json_config> --val-chcek-interval <evaluate_model_each_n_iterations> --output-dir <folder_to_store_ckpts> --log-path <path_to_store_json_log_file> --ckpt <path_to_ckpt> --eval
```

#### Convert a Model to the ONNX\* and OpenVINO™ format

PyTorch to ONNX:
```bash
python3 train.py --cfg <path_to_json_config> --ckpt <path_to_ckpt> --to-onnx --onnx-path <path_to_output_onnx>
```

ONNX to OpenVINO™:
```bash
mo.py --input_model <path_to_output_onnx>'
```

## Demo

You can try your models after converting them to the OpenVINO™ format or a [pretrained model from OpenVINO™](https://docs.openvinotoolkit.org/latest/machine_translation.html) using the [demo application from OpenVINO™ toolkit](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_machine_translation_demo_README.html)
