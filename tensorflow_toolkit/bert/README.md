# Bert base XNLI and Squad 1.1

BERT (Bidirectional Encoder Representations from Transformers) is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
XNLI is the Bert-base Chinese simplified and traditional sentence classification task with 12-layer, 768-hidden, 12 heads, 110M parameters.
Squad1.1 is the Bert-base questions answering task with the squad dataset of v1.1

Here we provided the steps for fine-tuning XNLI and Squad1.1 NLP tasks from the [Google Bert](https://github.com/google-research/bert) pre-trained model.

## Setup

### Prerequisites

   * Ubuntu 18.02
   * Python 3.6
   * Tensorflow 1.11.0
   * OpenVINO 2020.1

### Installation

1. Upload bert as git submodule 
    ```bash
    git submodule update --init --depth 1 $(git rev-parse --show-toplevel)/external/bert 
    ```

2. Create virtual environment
    ```bash
    cd $(git rev-parse --show-toplevel)/tensorflow_toolkit/bert
    virtualenv venv -p python3 --prompt="(bert)"
    ```

3. Activate virtual environment and set up OpenVINO™ variables:
    ```bash
    echo ". /opt/intel/openvino/bin/setupvars.sh" >> venv/bin/activate
    echo "export PYTHONPATH=\$PYTHONPATH:$(git rev-parse --show-toplevel)/external/bert" >> venv/bin/activate
    . venv/bin/activate
    ```

4. Install the tensorflow\*
    ```bash
    pip3 install -r requirements.txt
    ```


## XNLI

### Fine-tuning

1. Download the pre-trained Bert-base chinese model
    ```bash
    wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    ```

2. Refer the [Fine-tuning Example section](https://github.com/google-research/bert/blob/master/multilingual.md) for the data set and fine-tuning.

    ```bash
    export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12
    export XNLI_DIR=/path/to/xnli_dataset
    python $(git rev-parse --show-toplevel)/external/bert/run_classifier.py \
      --task_name=XNLI \
      --do_train=true \
      --data_dir=$XNLI_DIR \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=5e-5 \
      --num_train_epochs=2.0 \
      --output_dir=/tmp/xnli_output/
    ```

### Evaluation

```Bash
python $(git rev-parse --show-toplevel)/external/bert/run_classifier.py \
  --task_name=XNLI \
  --do_eval=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/tmp/xnli_output/model.ckpt-24543 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/tmp/xnli_output/
```

### Results 

```Bash
Accuracy = 0.774116
```

### Export to OpenVino IR

```Bash
python xnli_export.py \
  --task_name=XNLI \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/tmp/xnli_output/model.ckpt-24543 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --output_dir=/tmp/xnli_output \
  --export_dir=/tmp/xnli_output/export_model
```

#### Frozen Graph

```Bash
$ python3 -m tensorflow.python.tools.freeze_graph \
  --input_saved_model_dir=/tmp/xnli_output/export_model/<saved model folder>/ \
  --output_graph=./bert_xnli_fp32_graph.pb \
  --output_node_names=loss/LogSoftmax
```

#### OpenVino Intermediate Representation

```Bash
$ python mo.py --framework=tf \
  --input='input_ids_1,input_mask_1,segment_ids_1' \
  --output='loss/LogSoftmax' \
  --input_model=fp32_model.pb \
  --output_dir=fp32/ \
  --input_shape=[1,128],[1,128],[1,128] \
  --log_level=DEBUG \
  --disable_nhwc_to_nch
```

## BERT-Base, Uncased with SQuAD 1.1

### Data-set

First download the Standford Question Answering Dataset(SQuAD1.1) to some directory $SQUAD_DIR

* [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
* [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

### Fine-tuning


1. Download the Pre-trained Bert base uncased model

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
```
   
2. Fine-tune the model

```Bash
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export SQUAD_DIR=/path/to/bert/squad1.1/data
python $(git rev-parse --show-toplevel)/external/bert/run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

### Evaluation

```Bash
python $(git rev-parse --show-toplevel)/external/bert/run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/tmp/squad_base/model.ckpt-14599 \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

The dev set predictions will be saved into a file called predictions.json in the output_dir:

```Bash
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /tmp/squad_base/predictions.json
```

### Results

```Bash
Accuracy Metrics
Exact_match: 81.17313150425733
F1: 88.49906696207893
```

### Export to OpenVino IR

Export the saved model from the trained checkpoint

```Bash
python squad_export.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/tmp/squad_base/model.ckpt-14599 \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
  --export_dir =/tmp/squad_base/export_model/
```

#### Frozen Graph

```Bash
python3 -m tensorflow.python.tools.freeze_graph \
  --input_saved_model_dir=/tmp/squad_base/exported_model/<saved model folder>/ \
  --output_graph=bert_squad_fp32_graph.pb \
  --output_node_names=unstack
```

#### OpenVino Intermediate Representation

```Bash
python mo.py --framework=tf \
  --input='input_ids_1,input_mask_1,segment_ids_1' \
  --output='unstack' \
  --input_model=bert_squad_fp32_graph.pb \
  --output_dir=fp32/ \
  --input_shape=[1,384],[1,384],[1,384] \
  --log_level=DEBUG \
  --disable_nhwc_to_nchw
```

### OpenVino Dataset Annotation and Accuracy check

```Bash
convert_annotation squad \
  --testing_file $SQUAD_DIR/dev-v1.1.json \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 64 \
  --lower_case True \
  -o /output/dir/

accuracy_check --config squad_accuracy_check.yaml
```

