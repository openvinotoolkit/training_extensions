#Integrating NNCF into Transformers
https://github.com/huggingface/transformers

This folder contains a git patch to enable NNCF-based quantization for XNLI and SQuAD training pipelines of the huggingface transformers repository. Tested on BERT and RoBERTa-MNLI only.

Instructions:
1. Apply the `0001-Modifications-for-NNCF-usage.patch` file to the huggingface transformers repository checked out at commit id: `fbc5bf10cfe4d4ca81f8daacc148b0abd51dda5a`

2. Install the `transformers` library and the example scripts from this patched repository as described in the documentation for the huggingface transformers repository.

3. To start quantization-aware fine-tuning of BERT, use the regular scripts and command line parameters for XNLI and SQuAD training, but with additional `--nncf-config <path_to_nncf_config>` parameter. 
The NNCF configs are also provided in the same patch. 
For instance, the following command line, while run from the huggingface repository root, will launch XNLI quantization fine-tuning of BERT while starting out from a previously trained non-quantized XNLI-finetuned BERT checkpoint:
`python examples/run_xnli.py --model_type bert --model_name_or_path <path_to_pretrained_xnli_bert_fp32> --language zh --train_language zh --do_train --do_eval --data_dir <path_to_xnli_dataset> --per_gpu_train_batch_size 48 --learning_rate 5e-5 --num_train_epochs 4.0 --max_seq_length 128 --output_dir xnli_output --save_steps 200 --nncf-config nncf_bert_config_xnli.json --per_gpu_eval_batch_size 48`

    Distributed multiprocessing is also supported, simply use the corresponding version of the command line in the huggingface transformers repository with the same additional `--nncf-config` parameter.

    Same for SQuAD:
    `python examples/run_squad.py --model_type bert --model_name_or_path <path_to_pretrained_squad_bert_fp32> --do_train --do_eval --do_lower_case --train_file <path_to_squad_dataset>/train-v1.1.json --predict_file <path_to_squad_dataset>/dev-v1.1.json --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir models/wwm_uncased_finetuned_squad/ --per_gpu_eval_batch_size=6 --per_gpu_train_batch_size=6 --save_steps=400 --nncf-config nncf_bert_config_squad.json`

    For RoBERTa-MNLI:
    `python examples/run_glue.py --model_type roberta --model_name_or_path <path_to_pretrained_roberta_fp32> --task_name mnli --do_train --do_eval --data_dir <path_to_glue_dataset>/MNLI --per_gpu_train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 3.0 --max_seq_length 128 --output_dir mnli_roberta_output --save_steps 400 --nncf-config nncf_roberta_config_mnli.json`
    
    For DistilBERT-SST-2:
    `python examples/run_glue.py --model_type distilbert --model_name_or_path <path_to_pretrained_distilbert_fp32> --task_name SST-2 --do_train --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 16 --learning_rate 5e-5 --num_train_epochs 3.0 --output_dir distilbert_yelp_tmp_model --data_dir --data_dir <path_to_glue_dataset>/SST-2 --save_steps 100000 --nncf-config nncf_distilbert_config_sst2.json`

    Note that in all cases the training hyperparameters might have to be adjusted to accomodate the hardware you have available.

4. While running with the `--nncf-config` option, the training scripts will output NNCF-wrapped model checkpoints instead of the regular ones. You may evaluate these checkpoints using the same command lines for training above, but with the`--do_train` key omitted. In order to export these checkpoints into ONNX format, further add `--to-onnx <path_to_output_onnx_file>` to your evaluation command line parameters.

#Current best results:

All models use as their baselines the checkpoints obtained with the scripts and command line parameters from the corresponding sections in the original repository documentation. While fine-tuning the quantized model, the hyperparameters were left unchanged, i.e. the difference in the training script invocation was limited to adding `--nncf-config` option and specifying the pre-trained baseline model as the starting point for quantization fine-tuning. For RoBERTa-MNLI, no baseline model finetuning was necessary since the `roberta-large-mnli` model pretrained on MNLI was already available for download.

Make sure that you are running evaluation on a single GPU, since the repository evaluation scripts give inconsistent results when running multi-GPU evaluation.

**BERT-XNLI**:

_Full-precision FP32 baseline model_ - bert-base-chinese, trained on the Chinese portion of XNLI - 77.68% accuracy when evaluated on the Chinese portion of XNLI test set.

_INT8 model (symmetric weights, asymmetric activations quantization)_ - 77.22% accuracy in the same evaluation conditions.


**BERT-SQuAD v1.1**:

_Full-precision FP32 baseline model_ - bert-large-uncased-whole-word-masking model, trained on SQuAD v1.1 - 93.21% F1, 87.2% EM on the dev set,

_INT8 model (symmetric weights, asymmetric activations quantization)_ - 92.36% F1, 85.58% EM on the dev set.


**RoBERTA-MNLI**:

_Full-precision FP32 baseline model_ - roberta-large-mnli, pre-trained on MNLI - 90.6% accuracy (matched), 90.1% accuracy (mismatched)

_INT8 model (asymmetrically quantized)_ - 89.25% accuracy (matched), 88.9% accuracy (mismatched)


**DistilBERT-SST-2**

_Full-precision FP32 baseline model_ - distilbert-base-uncased-finetuned-sst-2-english, pre-trained on SST-2 - 91.1% accuracy

_INT8 model (symmetrically quantized)_ - 90.3% accuracy

