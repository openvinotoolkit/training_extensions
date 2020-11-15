# Question Answering

This is implementation of the train scripts in PyTorch\*.
The scripts allow to fine tune, distill and quantize pretrained [Transformers](https://github.com/huggingface/transformers) BERT-Large model for two question answering (QA) tasks:
1. Find answer's start stop positions for given question in in given context.
2. Calculate emebdding vectors for questions and contexts to fast find context with answer to given question

For details about the original model, check out
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805),
[HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771).

## Table of Contents

1. [Requirements](#requirements)
2. [Preparation](#preparation)
3. [Train QA model](#train-qa)
4. [Train Embedding model](#train-embedding)
5. [Convert a Model to OpenVINO™ format for Demo](#convert-a-models-to-openvino-format-for-demo)


## Requirements

To run scrips the Python\* 3.6 has to be installed on machine.
To install the required packages, run the following:

```bash
pip install -r requirements.txt
```

## Preparation

To finetune BERT for QA task the SQUAD 1.1 dataset is used. Please download SQUAD and unpack.
The dataset should have 3 files:
1. train-v1.1.json that contains train data
2. dev-v1.1.json that contains validation data
3. evaluate-v1.1.py that contains code to evaluate result
The folder that contains all these file will ber refered as ${SQUAD} below

## Train QA

After you prepared the data, you can train the model.
The train process consist from 3 steps:
1. Fine tune pretrained BERT-Large model for QA task by train_qa.py script.
2. Distill the fine tuned model to much smaller model by pack_and_distill.py script
3. Quantize the small model to INT8 by train_qa.py script.

### Fine tune BERT-Large for QA task

On this step [Transformers](https://github.com/huggingface/transformers) BertForQuestionAnswering model is initialized
by the pretrained bert-large-uncased-whole-word-masking model
and finetuned using SQUAD train dataset to model that allows to find answer start and stop position
for given question and context with answer.

To do this you may use the followed command line:

```bash
python3 train_qa.py \
--freeze_list=embedding \
--supervision_weight=0 \
--model_student=bert-large-uncased-whole-word-masking \
--output_dir=models/bert-large-uncased-wwm-squad-qa-fp32 \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--squad_eval_script=${SQUAD}/evaluate-v1.1.py \
--learning_rate=3e-5 \
--num_train_epochs=2 \
--max_seq_length_q=64 \
--max_seq_length_c=384 \
--per_gpu_eval_batch_size=16   \
--per_gpu_train_batch_size=2   \
--total_train_batch_size=48

```

As result the finetuned model has to be located in newly created folder 'models/bert-large-uncased-wwm-squad-qa-fp32'.

### Distill the fine tuned model to much smaller model

On this step the model from previous step can be packed by reducing number of layers, hidden size, self attention heads and replacing some block to more efficient
To do this you may use the followed command line:

```bash
python3 pack_and_distill.py \
--model_student=models/bert-large-uncased-wwm-squad-qa-fp32 \
--model_teacher=models/bert-large-uncased-wwm-squad-qa-fp32 \
--output_dir=models/bert-small-uncased-wwm-squad-qa-fp32 \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--squad_eval_script=${SQUAD}/evaluate-v1.1.py \
--pack_cfg=num_hidden_layers:12,ff_iter_num:4,num_attention_heads:8,hidden_size:512,pack_emb:1,hidden_act:orig \
--loss_weight_alpha=1.5 \
--learning_rate=5e-4 \
--learning_rate_for_tune=10e-4 \
--num_train_epochs=16 \
--max_seq_length_q=64 \
--max_seq_length_c=384 \
--per_gpu_eval_batch_size=32 \
--per_gpu_train_batch_size=4 \
--total_train_batch_size_for_tune=64 \
--total_train_batch_size=32

```

As result the packed small model has to be located in newly created folder 'models/bert-small-uncased-wwm-squad-qa-fp32'.

### Quantize the small packed model to INT8

On this step the model from previous step can be quantized to INT8 using [NNCF](https://github.com/openvinotoolkit/nncf) tool.
To do this you may use the followed command line:

```bash
python train_qa.py \
--freeze_list=none \
--supervision_weight=0.02 \
--kd_weight=1 \
--model_student=models/bert-small-uncased-wwm-squad-qa-fp32 \
--model_teacher=models/bert-large-uncased-wwm-squad-qa-fp32 \
--output_dir=models/bert-small-uncased-wwm-squad-int8 \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--squad_eval_script=${SQUAD}/evaluate-v1.1.py \
--learning_rate=1e-4 \
--num_train_epochs=16 \
--max_seq_length_q=64 \
--max_seq_length_c=384 \
--nncf_config=nncf_config.json \
--per_gpu_eval_batch_size=16 \
--per_gpu_train_batch_size=8 \
--total_train_batch_size=32

```

As result the packed small int8 model has to be located in newly created folder 'models/bert-small-uncased-wwm-squad-int8'.


## Train Embedding

This model allows to calculate embedding vectors for questions and contexts. The L2 distance from question embeddings can be measured to several context embeddings to find the best candidate with answer.

After you prepared the data, you can train the model.
The train process consist from 3 steps:
1. Fine tune pretrained BERT-Large model for Embedding task by train_qcemb.py script.
2. Distill the fine tuned model to much smaller model by pack_and_distill.py script
3. Quantize the small model to INT8 by train_qcemb.py script.

### Fine tune BERT-Large for Embedding task

On this step BertModelEMB model is initialized by the pretrained bert-large-uncased-whole-word-masking model
and finetuned using SQUAD train dataset to model that produces embeddings for question or context.
The L2 distance from question embeddings can be measured to several context embeddings to find the best candidate with answer.

To tune the embedding model you may use the followed command line:

```bash
python train_qcemb.py \
--freeze_list=embedding \
--supervision_weight=0.02 \
--model_teacher=bert-large-uncased-whole-word-masking \
--model_student=bert-large-uncased-whole-word-masking \
--output_dir=models/bert-large-uncased-wwm-squad-emb-fp32 \
--hnm_batch_size=8 \
--hnm_hist_num=32 \
--hnm_num=256 \
--loss_cfg=triplet_num:1,emb_loss:none \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--learning_rate=3e-5 \
--num_train_epochs=4 \
--max_seq_length_q=32 \
--max_seq_length_c=384 \
--per_gpu_eval_batch_size=16   \
--per_gpu_train_batch_size=2   \
--total_train_batch_size=32
```

As result the finetuned embeding model has to be located in newly created folder 'models/bert-large-uncased-wwm-squad-emb-fp32'.

### Distill the fine tuned embedding model to much smaller model

On this step the model from previous step can be packed by reducing number of layers, hidden size, self attention heads and replacing some block to more efficient
To do this you may use the followed command line:

```bash
python pack_and_distill.py \
--model_student=models/bert-large-uncased-wwm-squad-emb-fp32 \
--model_teacher=models/bert-large-uncased-wwm-squad-emb-fp32 \
--output_dir=models/bert-small-uncased-wwm-squad-emb-fp32 \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--squad_eval_script=${SQUAD}/evaluate-v1.1.py \
--pack_cfg=num_hidden_layers:12,ff_iter_num:4,num_attention_heads:8,hidden_size:512,pack_emb:1,hidden_act:orig \
--loss_weight_alpha=1.5 \
--learning_rate=3e-4 \
--learning_rate_for_tune=3e-4 \
--num_train_epochs=16 \
--max_seq_length_q=32 \
--max_seq_length_c=384 \
--per_gpu_eval_batch_size=32 \
--per_gpu_train_batch_size=4 \
--total_train_batch_size_for_tune=32 \
--total_train_batch_size=32
```

As result the packed small embedding model has to be located in newly created folder 'models/bert-small-uncased-wwm-squad-emb-fp32'.

### Quantize the small packed embeding model to INT8

On this step the embedding model from previous step can be quantized to INT8 using [NNCF](https://github.com/openvinotoolkit/nncf) tool.
To do this you may use the followed command line:

```bash
python train_qcemb.py \
--nncf_config=nncf_config.json \
--freeze_list=none \
--supervision_weight=0.02 \
--model_teacher=models/bert-large-uncased-wwm-squad-emb-fp32 \
--model_student=models/bert-small-uncased-wwm-squad-emb-fp32 \
--output_dir=models/bert-small-uncased-wwm-squad-emb-int8 \
--hnm_batch_size=8 \
--hnm_hist_num=32 \
--hnm_num=256 \
--loss_cfg=triplet_num:1,emb_loss:L2 \
--squad_train_data=${SQUAD}/train-v1.1.json \
--squad_dev_data=${SQUAD}/dev-v1.1.json \
--learning_rate=3e-5 \
--num_train_epochs=16 \
--max_seq_length_q=32 \
--max_seq_length_c=384 \
--per_gpu_eval_batch_size=16   \
--per_gpu_train_batch_size=8   \
--total_train_batch_size=32
```

As result the packed small int8 embedding model has to be located in newly created folder 'models/bert-small-uncased-wwm-squad-emb-int8'.




## Convert a Models to OpenVINO™ format for Demo

Each script together with pytorch model also save ONNX\* model into the same output folder. This ONNX\* model can be converted to OpenVINO™ format

```bash
mo.py --input_model <path_to_output_onnx>
```

After conversion them to the OpenVINO™ format you can try models using
the [demo for QA models](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_bert_question_answering_demo_README.html) or
the [demo for Embedding models](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_bert_question_answering_embedding_demo_README.html)

