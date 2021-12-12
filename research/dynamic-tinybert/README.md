<!--
Apache v2 license
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Dynamic-TinyBERT

This is the official Pytorch implementation of [**Dynamic-TinyBERT**](https://arxiv.org/pdf/2111.09645.pdf).
For detailed information about the method, please refer to our paper.

Our code is based on [Length Adaptive Transformer](https://github.com/clovaai/length-adaptive-transformer)'s work.
Currently, it supports BERT transformer and SQuAD 1.1 benchmark.


## Training


### Requirements
- Python 3
- PyTorch
- 🤗 Transformers
- torchprofile (to measure FLOPs)
- SigOpt


### Downloads
- SQuAD 1.1: Download following files in a `$SQUAD_DIR` directory:
[train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), and [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py).
- General (pre-trained) TinyBERT: Download from https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT 


### Step 1: Finetuning Pretrained Transformer (Teacher)
```
python run_squad.py \
--data_dir $SQUAD_DIR \
--model_type bert \
--model_name_or_path bert-base-uncased \
--do_train \
--do_eval \
--evaluate_during_training \
--save_only_best \
--do_lower_case \
--train_file train-v1.1.json \
--predict_file dev-v1.1.json \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 32 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR/fintuned_bert
```

### Step 2: Intermediate-layer Distillation of Teacher to General-TinyBERT Student
```
python run_squad.py \
--data_dir $SQUAD_DIR \
--model_type bert \
--model_name_or_path $GENERAL_TINYBERT_DIR \
--teacher_model_type bert \
--teacher_model_name_or_path $OUTPUT_DIR/fintuned_bert/checkpoint-best \
--do_train \
--evaluate_during_training \
--save_only_best \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 20 \
--max_seq_length 384 \
--doc_stride 128 \
--gradient_accumulation_steps 1 \
--state_loss_ratio 1.0 \
--att_loss_ratio 1.0 \
--att_mse \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR/intermediate

```

### Step 3: Prediction-layer Distillation of Teacher to Distilled TinyBERT Student

```
python run_squad.py \
--data_dir $SQUAD_DIR \
--model_type bert \
--model_name_or_path $OUTPUT_DIR/intermediate/checkpoint-last \
--teacher_model_type bert \
--teacher_model_name_or_path $OUTPUT_DIR/fintuned_bert/checkpoint-best \
--do_train \
--evaluate_during_training \
--save_only_best \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--max_seq_length 384 \
--doc_stride 128 \
--gradient_accumulation_steps 1 \
--state_loss_ratio 0.0 \
--att_loss_ratio 0.0 \
--pred_distill \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR/dynamic_tinybert
```


### Step 4 (Optional): Training with LengthDrop
Our paper shows that, overall, training Dynamic-TinyBERT further with LengthDrop does not add significant improvement.

```
python run_squad.py \
--data_dir $SQUAD_DIR \
--model_type bert \
--model_name_or_path $OUTPUT_DIR/dynamic_tinybert/checkpoint-best \
--teacher_model_type bert \
--teacher_model_name_or_path $OUTPUT_DIR/fintuned_bert/checkpoint-best \
--do_train \
--evaluate_during_training \
--save_only_best \
--do_lower_case \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--max_seq_length 384 \
--doc_stride 128 \
--gradient_accumulation_steps 1 \
--state_loss_ratio 0.0 \
--att_loss_ratio 0.0 \
--pred_distill \
--length_adaptive \
--num_sandwich 2 \
--length_drop_ratio_bound 0.2 \
--layer_dropout_prob 0.2 \
--overwrite_output_dir \
--output_dir $OUTPUT_DIR/dynamic_tinybert_ld_naive

```

### Step 5: Hyperparameter Optimization over Length Configurations

We use SigOpt which is a leading Bayesian Optimization software service provider to optimize length configurations for any possible target computational budget.
The steps to execute the SigOpt search are:

1. Install the sigopt python modules with pip install sigopt
2. Sign up for an account at https://sigopt.com. In order to use the API, you'll need your API token from the API tokens page.
3. Edit the parameters in the run-sigopt-search.sh script (model/data paths)
4. execute:
```
./run-sigopt-search.sh
```

The results are saved in a CSV format.