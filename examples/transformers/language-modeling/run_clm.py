#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch

import model_compression_research as mcr

import dataset_processing as preprocess


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    asym_softmax: bool = field(
        default=False,
        metadata={
            "help": "Whether to use asym quantization on softmax output"
        }
    )
    use_cache: bool = field(
        default=False
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
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
        default=None,
        metadata={"help": "List of dataset_name:dataset_config"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    data_process_type: str = field(
        default="concatenate_clm",
        metadata={
            "help": f"The preprocess method to use for data preprocessing. Choose from list: {list(preprocess.PROCESS_TYPE.keys())}"
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    dataset_cache_directory: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory where the processed dataset will be saved"}
    )
    add_gpt2_post_process: bool = field(
        default=False,
        metadata={
            "help": (
                "During tokenization, will add the <end_of_text> token as BOS and EOS."
            )
        },
    )

    def __post_init__(self):
        if self.datasets_name_config is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DistillationArguments))
    mcr.add_pruning_arguments_to_parser(parser)
    mcr.add_quantization_arguments_to_parser(parser)
    parser.add_argument('--lr_rewind', action='store_true', default=False)
    parser.add_argument('--rewind_slope_factor', type=float, default=1., help="Rewinding learning rate slope factor")
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, distill_args, compression_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, distill_args, compression_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

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
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

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
    
    # Quantization
    model_class = mcr.quantization_model_or_class_factory(compression_args, cls=AutoModelForCausalLM.from_config(config).__class__)
    if model_args.model_name_or_path:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = model_class.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    if compression_args.do_quantization and model_args.asym_softmax:
        logger.info('****ASYM SOFTMAX*****')
        def handle_value_matmul(module):
            if hasattr(module, "value_matmul"):
                module.value_matmul.input_fake_quant = mcr.default_activation_asym_fake_quant()
        model.apply(handle_value_matmul)
    if model_args.use_cache:
        logger.info("************* cache ************")
        model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))

    with training_args.main_process_first(desc="Dataset load and preprocess"):
        lm_datasets = preprocess.data_process(tokenizer, data_args, is_world_process_zero=is_main_process(training_args.local_rank))


    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    if training_args.do_predict:
        if "test" not in lm_datasets:
            raise ValueError()
        test_dataset = lm_datasets['test']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(test_dataset), data_args.max_eval_samples)
            test_dataset = test_dataset.select(range(max_eval_samples))


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Pruning
    pruning_callback = mcr.HFTrainerPruningCallback(compression_args)

    # Distillation
    if distill_args.distill:
        logger.info("*** Performing distillation ***")
        logger.info(
            "*** Initializing teacher from {} ***".format(distill_args.teacher_name_or_path))
        logger.info("*** Distillation parameters: {}".format(distill_args))
        # Initialize teacher model
        teacher = AutoModelForCausalLM.from_pretrained(
            distill_args.teacher_name_or_path,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # Handle logits names
        logit_names = ["logits"]
        weight = None
        # Handle cross_model_distillation
        hidden_alpha = None
        attention_alpha = None
        similarity_loss = 'mse'
        # TODO:
        """
        Should deep dive and understand better what's going on here.
        Should I change for example 'distilbert' to 'distilgpt'?
        I think not, seems like distilbert refers to some method.
        """
        if distill_args.cross_model_distillation is not None:
            from collections import defaultdict
            if distill_args.cross_model_distillation == 'distilbert':
                hidden_alpha = defaultdict(lambda: 0.)
                hidden_alpha[model.config.num_hidden_layers] = 1 - distill_args.cross_entropy_alpha - distill_args.knowledge_distillation_alpha
                similarity_loss = 'cosine_embedding'
            if distill_args.cross_model_distillation == 'tinybert':
                attention_alpha = defaultdict(lambda: 1.)
                hidden_alpha = defaultdict(lambda: 1.)
        logger.info("*** Applying teacher to student ***")
        model = mcr.hf_add_teacher_to_student(
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
        logger.info("*** Initialized a student and teacher for distillation successfully ***")
    
    # Rewinding
    if compression_args.do_prune and compression_args.lr_rewind:
        pruning_config = pruning_callback.config
        lr_sched_linear = lambda optimizer, num_warmup_steps, num_training_steps: mcr.get_linear_rewinding_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps, 
            pruning_config.policy_begin_step, 
            pruning_config.policy_end_step,
            pruning_config.pruning_frequency,
            compression_args.rewind_slope_factor,
         )
        lr_sched_cosine = lambda optimizer, num_warmup_steps, num_training_steps: mcr.get_cosine_rewinding_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps, 
            pruning_config.policy_begin_step, 
            pruning_config.policy_end_step,
            pruning_config.pruning_frequency,
         )
        # TODO Temp hack, need to find a way to use learning rate rewinding without overriding existing schedulers
        transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION[transformers.SchedulerType('linear')] = lr_sched_linear
        transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION[transformers.SchedulerType('cosine')] = lr_sched_cosine


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        callbacks=[pruning_callback],
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
            mcr.hf_remove_teacher_from_student(trainer.model)
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
            torch.save([vars(a) for a in [training_args, data_args, model_args, distill_args, compression_args]], os.path.join(training_args.output_dir, "args.bin"))
        except:
            logger.info("Failed to save arguments")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict/Test
    if training_args.do_predict:
        logger.info("*** Predict/Test ***")

        metrics = trainer.predict(test_dataset).metrics

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_eval_samples, len(test_dataset))
        try:
            perplexity = math.exp(metrics["test_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()