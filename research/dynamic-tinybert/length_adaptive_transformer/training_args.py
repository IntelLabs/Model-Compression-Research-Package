# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# 
# This file is copied from https://github.com/clovaai/length-adaptive-transformer


# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

from dataclasses import dataclass
from typing import Optional

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class TrainingArguments(HfTrainingArguments):
    warmup_ratio: Optional[float] = None
    logging_steps: int = 100
    evaluate_during_training: bool = True
    save_only_best: bool = True
    disable_tqdm: Optional[bool] = True
