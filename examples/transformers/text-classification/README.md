<!-- 
Apache v2 license
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier: Apache-2.0
 -->

# Text Classification with HuggingFace Transformers
Training script [`run_glue.py`](https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/text-classification/run_glue.py) based on Transformers v4.6.1 by HuggingFace

## Data
This script can be used to train models for GLUE datasets.
The datasets are downloaded and processed using `🤗/datasets` package.

## Usage
The script `run_glue.py` can be used for either training or inference of `🤗/transformers` models.

### Training
Train `bert-base-uncased` for MNLI dataset from GLUE Benchmark using the following command:

```bash
python run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name mnli \
    --do_train \
    --max_seq_length 128 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --output_dir <OUTPUT_DIR>
```

To prune the `bert-base-uncased` using iterative unstructured magnitude pruning to 85% sparsity add the following flags:
```bash
    --do_prune \
    --pruning_config config/iterative_unstructured_magnitude_85_config.json \
```

To enable learning rate rewinding also add `--lr_rewind`.

Add model distillation from an already trained teacher add the following flags:
```bash
    --distill \
    --teacher_name_or_path <TRAINED_MODEL> \
    --cross_entropy_alpha 0.5 \
    --knowledge_distillation_alpha 0.5 \
    --temperature 2.0 \
```

### Inference
Infer an already trained model on MNLI dataset using the following command:

```bash
python run_glue.py \
    --model_name_or_path <TRAINED_MODEL> \
    --task_name mnli \
    --do_eval \
    --max_seq_length 128 \
    --output_dir <OUTPUT_DIR>
```

### Quantization
Train with quantization aware training similar to [Q8BERT: Quantized 8Bit BERT](https://arxiv.org/abs/1910.06188) or perform inference of an already quantized model by adding the following flags:

```bash
  --do_quantization \
  --quantization_config config/quantization_config.json \
```
