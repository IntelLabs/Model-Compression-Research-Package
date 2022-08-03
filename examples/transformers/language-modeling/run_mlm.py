#!/usr/bin/env python
# coding=utf-8

# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
"""
This script is based on HuggingFace/transformers example: https://github.com/huggingface/transformers/blob/v4.6.1/examples/pytorch/language-modeling/run_mlm.py
Changes made to the script:
 1. Added pruning capabilities
 2. Added model distillation capabilities
 3. Added learning rate rewinding option
 4. Added methods to save all hyper-parameters used
 5. Removed pre-processing code and exported it to dataset_processing.py
"""

import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

import torch

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.optimization import get_scheduler

from model_compression_research import (
    add_pruning_arguments_to_parser,
    HFTrainerPruningCallback,
    hf_add_teacher_to_student,
    hf_remove_teacher_from_student,
    get_linear_rewinding_schedule_with_warmup,
)

import dataset_processing as preprocess


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_wwm: bool = field(
        default=False,
        metadata={"help": "Use Whole Word Masking DataCollator"}
    )


@dataclass
class DistillationArguments:
    distill: bool = field(
        default=False,
        metadata={
            "help": "Perform model distill"
        }
    )
    teacher_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to distillation teacher"
        }
    )
    cross_entropy_alpha: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Factor to be multiplied by with the cross entropy loss before combining all losses"
        }
    )
    knowledge_distillation_alpha: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Factor to be multiplied by with the KD loss before combining all losses"
        }
    )
    cross_model_distillation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether to use distilbert or tinybert distillation method"
        }
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Temperature of the softmax in the knowledge distillation loss computation"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datasets_name_config: Optional[List[str]] = field(
        default=None, metadata={"help": "The name:config list of datasets to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    data_process_type: str = field(
        default="concatenate",
        metadata={
            "help": f"The preprocess method to use for data preprocessing. Choose from list: {list(preprocess.PROCESS_TYPE.keys())}"
        },
    )
    short_seq_probability: float = field(
        default=0.1,
        metadata={
            "help": "The probability to parse document to shorter sentences than max_seq_length. Defaults to 0.1."
        },
    )
    nsp_probability: float = field(
        default=0.5,
        metadata={
            "help": "The probability to choose a random sentence when creating next sentence prediction examples."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep in memory the loaded dataset. Defaults to False."
        },
    )
    dataset_seed: int = field(
        default=42,
        metadata={
            "help": "Seed to use in dataset processing, different seeds might yield different datasets. This seed and the seed in training arguments are not related"
        },
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory where the processed dataset will be saved. If path exists, try to load processed dataset from this path."
        }
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DistillationArguments))
    add_pruning_arguments_to_parser(parser)
    parser.add_argument('--lr_rewind', action='store_true', default=False)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, distill_args, pruning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, distill_args, pruning_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    model_cls = AutoModelForPreTraining if data_args.data_process_type == 'segment_pair_nsp' else AutoModelForMaskedLM
    if model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    if not is_main_process(training_args.local_rank):
        torch.distributed.barrier()
    tokenized_datasets = preprocess.data_process(tokenizer, data_args)
    if training_args.local_rank != -1 and is_main_process(training_args.local_rank):
        torch.distributed.barrier()

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    if model_args.use_wwm:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    logger.info("Using data collator of type {}".format(data_collator.__class__.__name__))
    
    # Pruning
    pruning_callback = HFTrainerPruningCallback(pruning_args)

    metric = None
    if data_args.data_process_type == 'segment_pair_nsp':
        def metric(eval_prediction):
            def accuracy(logits, labels):
                preds = np.argmax(logits, axis=1)
                return (preds == labels[1]).mean()
            
            def loss(logits, labels):
                return torch.nn.CrossEntropyLoss()(torch.tensor(logits).view(-1, 2), torch.tensor(labels[1]).view(-1)).item()

            return {
                "nsp_acc": accuracy(eval_prediction.predictions, eval_prediction.label_ids),
                "nsp_loss": loss(eval_prediction.predictions, eval_prediction.label_ids),
            }
        model.config.keys_to_ignore_at_inference = ['prediction_logits']
        training_args.label_names = ['labels', 'next_sentence_label']

    # Distillation
    if distill_args.distill:
        logger.info("*** Performing distillation ***")
        logger.info(
            "*** Initializing teacher from {} ***".format(distill_args.teacher_name_or_path))
        logger.info("*** Distillation parameters: {}".format(distill_args))
        # Initialize teacher model
        teacher = model_cls.from_pretrained(
            distill_args.teacher_name_or_path,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # Handle logits names
        if data_args.data_process_type == 'segment_pair_nsp':
            logit_names = ['prediction_logits', 'seq_relationship_logits']
            weight = [1, 1]
        else:
            logit_names = ["logits"]
            weight = None
        # Handle cross_model_distillation
        hidden_alpha = None
        attention_alpha = None
        similarity_loss = 'mse'
        if distill_args.cross_model_distillation is not None:
            if distill_args.cross_model_distillation == 'distilbert':
                hidden_alpha = defaultdict(lambda: 0.)
                hidden_alpha[model.config.num_hidden_layers] = 1 - distill_args.cross_entropy_alpha - distill_args.knowledge_distillation_alpha
                similarity_loss = 'cosine_embedding'
            if distill_args.cross_model_distillation == 'tinybert':
                attention_alpha = defaultdict(lambda: 1.)
                hidden_alpha = defaultdict(lambda: 1.)
        logger.info("*** Applying teacher to student ***")
        model = hf_add_teacher_to_student(
            model,
            teacher,
            student_alpha=distill_args.cross_entropy_alpha,
            teacher_ce_alpha=distill_args.knowledge_distillation_alpha,
            teacher_hidden_alpha=hidden_alpha,
            teacher_attention_alpha=attention_alpha,
            teacher_similarity_loss=similarity_loss,
            teacher_ce_temperature=distill_args.temperature,
            teacher_logit_names=logit_names,
            teacher_ce_weights=weight,
        )

    # Rewinding
    if pruning_args.do_prune and pruning_args.lr_rewind:
        pruning_config = pruning_callback.config
        lr_sched_fn = lambda optimizer, num_warmup_steps, num_training_steps: get_linear_rewinding_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps, 
            pruning_config.policy_begin_step, 
            pruning_config.policy_end_step,
            pruning_config.pruning_frequency,
         )
        # TODO Temp hack, need to find a way to use learning rate rewinding without overriding existing schedulers
        transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION[transformers.SchedulerType('linear')] = lr_sched_fn
    

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[pruning_callback],
        compute_metrics=metric,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if distill_args.distill:
            hf_remove_teacher_from_student(trainer.model)
            # There might have been a change in the forward signature of the model
            # This hack will reset the signature saved in the trainer
            trainer._signature_columns = None
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        try:
            torch.save([vars(a) for a in [training_args, data_args, model_args, distill_args, pruning_args]], os.path.join(training_args.output_dir, "args.bin"))
        except:
            logger.info("Failed to save arguments")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        if "eval_nsp_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"] - metrics["eval_nsp_loss"])
            except:
                logger.warning("Perplexity computation failed")
                perplexity = math.exp(metrics["eval_loss"])
        else:
                perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
