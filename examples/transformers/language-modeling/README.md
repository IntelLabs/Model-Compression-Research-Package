<!-- 
Apache v2 license
Copyright (C) 2021 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Masked Language-Modeling with HuggingFace Transformers
Training script [`run_mlm.py`](https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/language-modeling/run_mlm.py) based on Transformers v4.6.1 by HuggingFace

## Data
Datasets are downloaded and processed using `🤗/datasets` package.

## Usage
The script `run_mlm.py` can be used for either training or inference of `🤗/transformers` models.

### Training
Train `bert-base-uncased` on English Wikipedia dataset using the following command:

``` bash
python run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --datasets_name_config wikipedia:20200501.en \
    --do_train \
    --data_process_type segment_pair_nsp \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --max_steps 100000 \
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

This script also supports distributed training. 
For example, pruning `bert-base-uncased` using the previous configuration while distilling knowledge from `bert-base-uncased` can be done using the following command:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<NUM_GPU> \
    --master_port=<PORT> \
    run_mlm.py \
        --model_name_or_path bert-base-uncased \
        --dataset_name_config wikipedia:20200501.en \
        --do_train \
        --data_process_type segment_pair_nsp \
        --dataset_cache_dir <DATA_CACHE_DIR> \
        --max_steps 100000 \
        --do_prune \
        --pruning_config config/iterative_unstructured_magnitude_85_config.json \
        --distill \
        --teacher_name_or_path bert-base-uncased \
        --cross_entropy_alpha 0.5 \
        --knowledge_distillation_alpha 0.5 \
        --temperature 2.0 \
        --output_dir <OUTPUT_DIR>
```

Sharded DDP training from [FairScale](https://github.com/facebookresearch/fairscale) is also supported by adding `--sharded_ddp simple` to the command.
Note: simple is the only mode supported in this package.

### Inference
Infer an already trained model on English Wikipedia dataset using the following command:
``` bash
python run_mlm.py \
    --model_name_or_path <TRAINED_MODEL> \
    --do_eval \
    --dataset_name_config wikipedia:20200501.en \
    --data_process_type segment_pair_nsp \
    --dataset_cache_dir <DATA_CACHE_DIR> \
    --output_dir <OUTPUT_DIR>
```
