#!/bin/bash
# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Teacher Preparation

# Notes:
# Auto mixed precision can be used by adding --fp16
# Distributed training can be used with the torch.distributed.lauch app

TEACHER_PATH=/tmp/bert-base-uncased-teacher-preparation
OUTPUT_DIR=$TEACHER_PATH
DATA_CACHE_DIR=/tmp/data-cache-dir

python ../../examples/transformers/language-modeling/run_mlm.py \
    --model_name_or_path bert-base-uncased \
    --datasets_name_config wikipedia:20200501.en \
    --data_process_type segment_pair_nsp \
    --dataset_cache_dir $DATA_CACHE_DIR \
    --do_train \
    --max_steps 100000 \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 256 \
    --output_dir $OUTPUT_DIR
