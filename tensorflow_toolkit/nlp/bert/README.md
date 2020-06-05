# Fine tuning Bert base XNLI and Squad 1.1 from Google Bert

## Prerequistes

	*Ubuntu 18.02
	*Tensorflow 1.1X
	*OpenVINO 2020.1

## TensorFlow BERT Installation

1. Clone the google bert

	```bash
	$ git clone https://github.com/google-research/bert
	$ cd bert
	```
2. This folder contains a git patch to add support to export the saved model from the trained model for xnli and squad.
   Apply the `0001-Add-support-to-export-the-saved-model-to-the-run_cla.patch` file to the google bert repository

	```bash
	$ git checkout eedf5716ce1268e56f0a50264a88cafad334ac61
	$ git am 0001-Add-support-to-export-the-saved-model-to-the-run_cla.patch
	```

### Fine Tuning BERT-Base with XNLI dataset

1. Download the pre-trained Bert-base chinese model

	```bash
	$ wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
	```
2. Refer the Fine-tuning Example section of the multilingual google wiki for the data set and finetuning

	```bash
	https://github.com/google-research/bert/blob/master/multilingual.md
	```
	```bash
	$ export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12
	$ export XNLI_DIR=/path/to/xnli
	$ pip install tensorflow==1.14
	```
	```bash
	$ python run_classifier.py \
	--task_name=XNLI \
	--do_train=true \
	--do_eval=true \
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
	With the Chinese-only model, the result:

	```bash
	eval_accuracy = 0.774116
	eval_loss = 0.83554
	global_step = 24543
	loss = 0.74603
	```
3. Export the saved model from the trained checkpoint

	```bash
	$python run_classifier.py \
	--task_name=XNLI \
	--do_eval=true \
	--data_dir=$XNLI_DI \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/model.ckpt-24543 \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=5e-5 \
	--num_train_epochs=2.0 \
	--output_dir=/tmp/xnli_output \
	--export_dir=/tmp/xnli_output/export_model
	```
4. Clone the tensorflow source and generate the frozen graph

	```bash
	$ git clone -b r1.14 --single-branch https://github.com/tensorflow/tensorflow.git
	$ cd tensorflow
	$ python tensorflow/python/tools/freeze_graph.py \
	--input_saved_model_dir=/tmp/xnli_output/export_model/<folder>/ \
	--output_graph= ./bert_xnli_fp32_graph.pb \
	--output_node_names=loss/LogSoftmax
	```
5. Run the OpenVino Model Optimizer to generate the IR

	```bash
	$ python $MO_DIR/mo.py --framework=tf \
	--input='input_ids_1,input_mask_1,segment_ids_1' \
	--output='loss/LogSoftmax' \
	--input_model=fp32_model.pb \
	--output_dir=fp32/ \
	--input_shape=[1,128],[1,128],[1,128] \
	--log_level=DEBUG --disable_nhwc_to_nch
	```

### Fine Tuning BERT-Base, Uncased with SQuAD 1.1 dataset

1. Download the Pre-trained Bert base uncased model
	```bash
	$ wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
	```

2.  Refer the SQuad1.1 section of the google wiki for the data set and finetuning

	```bash
	Download the below data to some directory $SQUAD_DIR
	$wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	$wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
	$wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
	```
	Fine-Tuning
	```bash
	$ pip install tensorflow==1.14
	$ export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
	$ export SQUAD_DIR=/path/to/bert/squad1.1/data
	```
	```bash
	$ python run_squad.py \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	--do_train=True \
	--train_file=$SQUAD_DIR/train-v1.1.json \
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
	```bash
	$ python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /tmp/squad_base/predictions.json
	```
	Results:
		{"exact_match": 81.17313150425733, "f1": 88.49906696207893}

3. Export the saved model from the trained checkpoint

	```bash
	$ python run_squad.py \
	--vocab_file=$BERT_BASE_DIR/vocab.txt \
	--bert_config_file=$BERT_BASE_DIR/bert_config.json \
	--init_checkpoint=$BERT_BASE_DIR/model.ckpt-14599 \
	--do_predict=True \
	--predict_file=$SQUAD_DIR/dev-v1.1.json \
	--train_batch_size=12 \
	--learning_rate=3e-5 \
	--num_train_epochs=2.0 \
	--max_seq_length=384 \
	--doc_stride=128 \
	--output_dir=/tmp/squad_base/
	--export_dir = /tmp/squad_base/export_model/
	```

4. Clone the tensorflow source and generate the frozen graph

	```bash
	$ git clone -b r1.14 --single-branch https://github.com/tensorflow/tensorflow.git
	$ cd tensorflow
	$ python tensorflow/python/tools/freeze_graph.py \
	--input_saved_model_dir=/tmp/squad_base/exported_model/<>/ \
	--output_graph= ./bert_squad_fp32_graph.pb \
	--output_node_names=unstack
	```

5. Run the OpenVino Model Optimizer to generate the IR

	```bash
	$ python mo.py --framework=tf \
	--input='input_ids_1,input_mask_1,segment_ids_1' \
	--output='unstack' \
	--input_model=bert_squad_fp32_graph.pb \
	--output_dir=fp32/ \
	--input_shape=[1,384],[1,384],[1,384] \
	--log_level=DEBUG \
	--disable_nhwc_to_nchw
	```

6. Run the OpenVino Dataset annotation conversion to generate the input for the accuracy checker

	```bash
	$ convert_annotation squad \
	--testing_file $SQUAD_DIR/dev-v1.1.json \
	--vocab_file $BERT_BASE_DIR/vocab.txt \
	--max_seq_length 384 \
	--doc_stride 128 \
	--max_query_length 64 \
	--lower_case True \
	-o /output/dir/
	```
7. Run the OpenVino accuracy checker to verify the dataset metrics

	```bash
	$ accuracy_check --config squad_accuracy_check.yaml
	```
