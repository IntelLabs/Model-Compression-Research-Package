# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# 
# This file is based on run_squad.py from: https://github.com/clovaai/length-adaptive-transformer
# with the addition of:
#   1. TinyBERT/distillation training
#   2. hyper-parameter search using sigopt

# coding=utf-8
# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0
#####
# Original code is from https://github.com/huggingface/transformers
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import torchprofile
from collections import defaultdict, OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import KLDivLoss, MSELoss, CosineSimilarity
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors import squad
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from length_adaptive_transformer.drop_and_restore_utils import (
    sample_length_configuration,
    sample_layer_configuration,
    add_drop_and_restore_args,
    add_search_args,
)
from length_adaptive_transformer.evolution import (
    Evolution, approx_ratio, inverse, store2str
)

from length_adaptive_transformer.modeling_bert import TinyBertForQuestionAnswering
from length_adaptive_transformer.modeling_roberta import TinyRobertaForQuestionAnswering

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def dump_metrics(result, eval_time, args, macs=0, rate=0.):
    report = OrderedDict(sorted(result.items()))
    report['eval_time'] = eval_time
    report['model'] = args.model_name_or_path
    report['macs'] = macs
    report['rate'] = rate
    with open(args.metrics_log, 'a') as f:
        json.dump(report, f)
        f.write('\n')


def convert_fmt(file):
    target = os.path.splitext(file)[0] + '.csv'
    with open(file) as f, open(target, 'w') as new_f:
        for i, line in enumerate(f.readlines()):
            line = json.loads(line.rstrip())
            if i == 0:
                new_f.write(','.join(line.keys()) + '\n')
            new_f.write(','.join([str(v) for v in line.values()]) + '\n')


def disabled_tqdm(it, *args, **kwargs):
    return it


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def div_loss(loss, args):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
    return loss


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tb_log)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    retention_params = []
    wd_params = []
    no_wd_params = []
    for n, p in model.named_parameters():
        if "retention" in n:
            retention_params.append(p)
        elif any(nd in n for nd in no_decay):
            no_wd_params.append(p)
        else:
            wd_params.append(p)
    optimizer_grouped_parameters = [
        {"params": wd_params, "weight_decay": args.weight_decay, "lr": args.learning_rate},
        {"params": no_wd_params, "weight_decay": 0.0, "lr": args.learning_rate}
    ]
    if len(retention_params) > 0:
        optimizer_grouped_parameters.append(
            {"params": retention_params, "weight_decay": 0.0, "lr": args.lr_soft_extract}
        )

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #     os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        logger.info("Training with fp16.")

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss_sum = 0.0
    loss_sum = defaultdict(float)
    best = {'f1': None}
    model.zero_grad()
    # Added here for reproductibility
    set_seed(args)

    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            inputs["output_attentions"] = args.length_config is not None

            num_h_layers = model.config.num_hidden_layers if hasattr(model, "config") else model.module.config.num_hidden_layers

            layer_config = sample_layer_configuration(
                num_h_layers,
                layer_dropout_prob=args.layer_dropout_prob,
                layer_dropout=0,
            )
            inputs["layer_config"] = layer_config

            inputs["length_config"] = args.length_config

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            task_loss = div_loss(outputs[0], args)
            loss_sum["full"] += task_loss.item()
            loss = task_loss
            if args.length_adaptive:
                loss = loss / (args.num_sandwich + 2)

            tr_loss_sum += loss.item()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # inplace distillation
            if args.length_adaptive:
                start_logits = outputs[1].detach()
                end_logits = outputs[2].detach()

                for i in range(args.num_sandwich + 1):
                    inputs["output_attentions"] = True


                    num_h_layers = model.config.num_hidden_layers if hasattr(model, "config") else model.module.config.num_hidden_layers

                    layer_config = sample_layer_configuration(
                        num_h_layers,
                        layer_dropout_prob=args.layer_dropout_prob,
                        layer_dropout=(args.layer_dropout_bound if i == 0 else None),
                        layer_dropout_bound=args.layer_dropout_bound,
                    )
                    inputs["layer_config"] = layer_config

                    length_config = sample_length_configuration(
                        args.max_seq_length,
                        num_h_layers,
                        layer_config,
                        length_drop_ratio=(args.length_drop_ratio_bound if i == 0 else None),
                        length_drop_ratio_bound=args.length_drop_ratio_bound,
                    )
                    inputs["length_config"] = length_config

                    outputs_sub = model(**inputs)
                    task_loss_sub = div_loss(outputs_sub[0], args)
                    if i == 0:
                        loss_sum["smallest"] += task_loss_sub.item()
                        loss_sum["sub"] += 0
                    else:
                        loss_sum["sub"] += task_loss_sub.item() / args.num_sandwich

                    start_logits_sub = outputs_sub[1]
                    end_logits_sub = outputs_sub[2]
                    loss_fct = KLDivLoss(reduction="batchmean")
                    start_kl_loss = loss_fct(F.log_softmax(start_logits, -1), F.softmax(start_logits_sub, -1))
                    end_kl_loss = loss_fct(F.log_softmax(end_logits, -1), F.softmax(end_logits_sub, -1))
                    loss = div_loss((start_kl_loss + end_kl_loss) / 2, args)
                    loss_sum["kl"] += loss.item() / (args.num_sandwich + 1)
                    loss = loss / (args.num_sandwich + 2)

                    tr_loss_sum += loss.item()
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                (step + 1) == len(train_dataloader) <= args.gradient_accumulation_steps
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    lr = scheduler.get_lr()[0]
                    loss = tr_loss_sum / args.logging_steps
                    tr_loss_sum = 0.0
                    logs = {"lr": lr, "loss": loss}
                    log_str = f"[{global_step:5d}] lr {lr:g} | loss {loss:2.3f}"

                    for key, value in loss_sum.items():
                        value /= args.logging_steps
                        loss_sum[key] = 0.0
                        logs[f"{key}_loss"] = value
                        log_str += f" | {key}_loss {value:2.3f}"

                    for k, v in logs.items():
                        tb_writer.add_scalar(k, v, global_step)
                    logger.info(log_str)

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, eval_time = evaluate(args, model, tokenizer, prefix="")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        msg = " | ".join([f"{k} {v:.2f}" for k, v in results.items()])
                        logger.info(f"  [{global_step:5d}] {msg}")

                    if args.save_only_best:
                        output_dirs = []
                    else:
                        output_dirs = [os.path.join(args.output_dir, f"checkpoint-{global_step}")]
                    if args.evaluate_during_training and (best['f1'] is None or results['f1'] > best['f1']):
                        logger.info("Congratulations, best model so far!")
                        output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))
                        best = {'step': global_step, 'f1': results['f1'], 'em': results['exact'], 'eval_time': eval_time}

                    for output_dir in output_dirs:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if 0 < args.max_steps <= global_step:
                break

        if 0 < args.max_steps <= global_step:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, best



def distill(args, train_dataset, teacher_model, student_model, tokenizer):
    """ Train the model with distillation
        Assumes that teacher and student models share same token embedding layer.
        So, the same data is loaded and fed to both teacher and student models.
        This function code is base on TinyBERT implementation 
        (https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).
    """

    ############################################################################################
    # no multi-node distributed trainig, continued training and fp16 support for KD
    ############################################################################################

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tb_log)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) # no multi-node
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:  # number of training steps = number of epochs * number of batches
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # layer numbers of teacher and student
    teacher_layer_num = teacher_model.config.num_hidden_layers if hasattr(teacher_model, "config") else teacher_model.module.config.num_hidden_layers
    student_layer_num = student_model.config.num_hidden_layers if hasattr(student_model, "config") else student_model.module.config.num_hidden_layers
    layers_per_block = int(teacher_layer_num / student_layer_num)

    # multi-gpu training
    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)

    # Prepare loss functions
    loss_mse = MSELoss()
    loss_cs = CosineSimilarity(dim=2)
    loss_cs_att = CosineSimilarity(dim=3)

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean()

    # Distill!
    logger.info("***** Running distillation training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss_sum = 0.0 # for LAT
    tr_loss, logging_loss = 0.0, 0.0
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    best = {'f1': None}
    student_model.zero_grad()
    train_iterator = range(epochs_trained, int(args.num_train_epochs))

    set_seed(args)  # Added here for reproductibility
    best_val_metric = None
    for epoch_n in train_iterator:
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            student_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(student_model, "config") and hasattr(student_model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
   
             # teacher model output
            teacher_model.eval() # set teacher as eval mode
            teacher_output_att = True if args.att_loss_ratio > 0.0 else False
            teacher_output_hid = True if args.state_loss_ratio > 0.0 else False
            with torch.no_grad():
                outputs_teacher = teacher_model(output_attentions=teacher_output_att, output_hidden_states=teacher_output_hid, **inputs)

            # setup input for the student, LAT
            inputs["output_attentions"] = args.length_config is not None or args.att_loss_ratio > 0.0
            inputs["output_hidden_states"] = args.state_loss_ratio > 0.0
            num_h_layers = student_model.config.num_hidden_layers if hasattr(student_model, "config") else student_model.module.config.num_hidden_layers

            layer_config = sample_layer_configuration(
                num_h_layers,
                layer_dropout_prob=args.layer_dropout_prob,
                layer_dropout=0,
            )
            inputs["layer_config"] = layer_config

            inputs["length_config"] = args.length_config

            # student model output
            outputs_student = student_model(is_student=True, **inputs)

           
            
            # Knowledge Distillation loss
            # 1) logits distillation
            if args.pred_distill:
                start_kd_loss = soft_cross_entropy(outputs_student[1], outputs_teacher[1])
                end_kd_loss = soft_cross_entropy(outputs_student[2], outputs_teacher[2])
                kd_loss = (start_kd_loss + end_kd_loss) / 2
                loss = kd_loss
                if args.length_adaptive:
                    loss = loss / (args.num_sandwich + 2)
                    tr_loss_sum += loss.item()
                tr_cls_loss += loss.item()

            # 2) embedding and last hidden state distillation
            if args.state_loss_ratio > 0.0:
                assert not args.pred_distill
                teacher_reps = outputs_teacher[3]
                student_reps = outputs_student[3]

                if args.partial_reps:
                    new_teacher_reps = [teacher_reps[0], teacher_reps[teacher_layer_num]]
                    new_student_reps = [student_reps[0], student_reps[student_layer_num]]
                else:
                    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_student_reps = student_reps

                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    # cosine similarity loss
                    if args.state_distill_cs:
                        tmp_loss = 1.0 - loss_cs(student_rep, teacher_rep).mean()
                    # MSE loss
                    else:
                        tmp_loss = loss_mse(student_rep, teacher_rep).mean()
                    rep_loss += tmp_loss
                loss = args.state_loss_ratio * rep_loss
                tr_rep_loss += rep_loss.item()

            # 3) Attentions distillation
            if args.att_loss_ratio > 0.0:
                assert not args.pred_distill and args.state_loss_ratio > 0.0
                teacher_atts = outputs_teacher[4]
                student_atts = outputs_student[4]

                assert teacher_layer_num == len(teacher_atts)
                assert student_layer_num == len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                              teacher_att)
                    if args.att_mse:
                        tmp_loss = loss_mse(student_att, teacher_att).mean()
                    else:
                        tmp_loss = 1.0 - loss_cs_att(student_att, teacher_att).mean()
                    att_loss += tmp_loss

                loss += args.att_loss_ratio * att_loss
                tr_att_loss += att_loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # back propagate
            loss.backward()

            # LAT
            # inplace distillation
            if args.length_adaptive:
                start_logits = outputs_student.start_logits.detach()
                end_logits = outputs_student.end_logits.detach()

                for i in range(args.num_sandwich + 1):
                    inputs["output_attentions"] = True
                    num_hidden = student_model.config.num_hidden_layers if hasattr(student_model, "config") else student_model.module.config.num_hidden_layers
                    layer_config = sample_layer_configuration(
                        num_hidden,
                        layer_dropout_prob=args.layer_dropout_prob,
                        layer_dropout=(args.layer_dropout_bound if i == 0 else None),
                        layer_dropout_bound=args.layer_dropout_bound,
                    )
                    inputs["layer_config"] = layer_config

                    length_config = sample_length_configuration(
                        args.max_seq_length,
                        num_hidden,
                        layer_config,
                        length_drop_ratio=(args.length_drop_ratio_bound if i == 0 else None),
                        length_drop_ratio_bound=args.length_drop_ratio_bound,
                    )
                    inputs["length_config"] = length_config

                    outputs_student_sub = student_model(is_student=True, **inputs)

                    start_logits_sub = outputs_student_sub.start_logits
                    end_logits_sub = outputs_student_sub.end_logits
                    loss_fct = KLDivLoss(reduction="batchmean")
                    start_kl_loss = loss_fct(F.log_softmax(start_logits, -1), F.softmax(start_logits_sub, -1))
                    end_kl_loss = loss_fct(F.log_softmax(end_logits, -1), F.softmax(end_logits_sub, -1))
                    loss = div_loss((start_kl_loss + end_kl_loss) / 2, args)
                    loss = loss / (args.num_sandwich + 2)

                    tr_loss_sum += loss.item()
                    loss.backward()

            if args.length_adaptive:
                tr_loss = tr_loss_sum
            else:
                tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")

            
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                (step + 1) == len(train_dataloader) <= args.gradient_accumulation_steps
            ):
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student_model.zero_grad()
                global_step += 1

                # change to evaluation mode
                student_model.eval()
                
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    lr = scheduler.get_lr()[0]
                    loss = tr_loss / args.logging_steps
                    tr_loss = 0.0
                    logs = {"lr": lr, "loss": loss}
                    log_str = f"[{global_step:5d}] lr {lr:g} | loss {loss:2.3f}"

                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    logs['cls_loss'] = cls_loss
                    logs['att_loss'] = att_loss
                    logs['rep_loss'] = rep_loss

                    for k, v in logs.items():
                        tb_writer.add_scalar(k, v, global_step)
                    logger.info(log_str)

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results, eval_time = evaluate(args, student_model, tokenizer, prefix="eval")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        msg = " | ".join([f"{k} {v:.2f}" for k, v in results.items()])
                        logger.info(f"  [{global_step:5d}] {msg}")

                    if args.save_only_best:
                        output_dirs = []
                    else:
                        output_dirs = [os.path.join(args.output_dir, f"checkpoint-{global_step}")]
                    if args.evaluate_during_training and (best['f1'] is None or results['f1'] > best['f1']):
                        logger.info("Congratulations, best model so far!")
                        output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))
                        best = {'step': global_step, 'f1': results['f1'], 'em': results['exact'], 'eval_time': eval_time}

                    for output_dir in output_dirs:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = student_model.module if hasattr(student_model, "module") else student_model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                
                # change student model back to train mode
                student_model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step



def evaluate(args, model, tokenizer, prefix="", dataset=None, examples=None, features=None):
    if dataset is None:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    if not args.do_search:
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            inputs["output_attentions"] = None
            inputs["layer_config"] = None
            inputs["length_config"] = None

            if args.length_config is not None:
                inputs["output_attentions"] = True
                inputs["length_config"] = args.length_config

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(outputs) >= 5:
                start_logits = to_list(outputs[0][i])
                start_top_index = to_list(outputs[1][i])
                end_logits = to_list(outputs[2][i])
                end_top_index = to_list(outputs[3][i])
                cls_logits = to_list(outputs[4][i])

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )
            else:
                start_logits = to_list(outputs[0][i])
                end_logits = to_list(outputs[1][i])
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_dir = os.path.join(args.output_dir, prefix)
    if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_dir)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results, evalTime


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            str(args.max_seq_length),
            str(args.doc_stride),
            str(args.max_query_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    add_drop_and_restore_args(parser)

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    add_search_args(parser)

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best", action="store_true", help="Save only when hit best validation score.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--use_gpuid", type=int, default=-1, help="Use a specific GPU only")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=10, help="multiple threads for converting example to features")
    parser.add_argument(
        "--teacher_model_type",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained teacher model or shortcut name",
    )
    parser.add_argument('--state_loss_ratio', type=float, default=0.0)
    parser.add_argument('--att_loss_ratio', type=float, default=0.0)
    parser.add_argument('--pred_distill', action='store_true')
    parser.add_argument("--state_distill_cs", action="store_true", help="If this is using Cosine similarity for the hidden and embedding state distillation. vs. MSE")
    parser.add_argument("--att_mse", action="store_true")
    parser.add_argument("--partial_reps", action="store_true")
    parser.add_argument("--measure_rate", default=None, type=str)
    parser.add_argument("--tb_log", default='tb_logs', type=str)
    parser.add_argument("--metrics_log", default='metrics.log', type=str)
    parser.add_argument("--do_optuna_search", action="store_true")
    parser.add_argument("--do_sigopt_search", action="store_true")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.use_gpuid > -1:
        device = args.use_gpuid
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    squad.tqdm = disabled_tqdm

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False
    )

    # TinyBERT model setup
    if args.att_loss_ratio > 0 or args.state_loss_ratio > 0 or args.pred_distill:
        model_class = TinyBertForQuestionAnswering
    else:
        model_class = AutoModelForQuestionAnswering

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    if args.do_search:
        if hasattr(model, "config"):
            model.config.output_attentions = True
        else:
            model.module.config.output_attentions = True

    if args.length_config is not None:
        args.length_config = eval(args.length_config)
    assert not (args.length_config and args.length_adaptive)
    if args.length_adaptive or args.do_search:
        args.max_seq_length = args.max_seq_length

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


    # Measure performence/Flops for given length-drop rates:
    if args.measure_rate is not None:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
        size = (1, args.max_seq_length)
        dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.zeros(size, dtype=torch.long).to(args.device),
        )
        rates = eval(args.measure_rate)
        num_hidden_layers = model.module.config.num_hidden_layers if hasattr(model, "module") else model.config.num_hidden_layers
        results = {}
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        out_file_path = os.path.join(args.output_dir, 'measure_out.log')
        with open(out_file_path, 'w') as out_file:
            for rate in rates:
                out_file.write(str(rate) + ":\n")
                length_config = sample_length_configuration(
                        args.max_seq_length,
                        num_hidden_layers,
                        layer_config=None,
                        length_drop_ratio=float(rate),
                        length_drop_ratio_bound=float(rate))
                args.length_config = length_config
                model.bert.set_length_config(length_config)

                if hasattr(model, "module"):
                    model.module.config.output_attentions = True
                else:
                    model.config.output_attentions = True

                macs = torchprofile.profile_macs(model, args=dummy_inputs)
                result, evalTime = evaluate(args, model, tokenizer, dataset=dataset, examples=examples, features=features)
                dump_metrics(result, evalTime, args, macs=macs, rate=rate)

                result = dict((k, v) for k, v in result.items())
                results.update(result)
                res_str = "sequences={}\nmacs={}\neval={}\n".format(length_config, macs, results)
                out_file.write(res_str)
                logger.info(res_str)
        return

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        # normal training (fine-tuning)
        if args.teacher_model_type is None or args.teacher_model_name_or_path is None:
            global_step, best = train(args, train_dataset, model, tokenizer)
            logger.info(f" global_step = {global_step} | best (step, em, f1, eval_time) = ({best['step']}, {best['em']:f}, {best['f1']:f}, {best['eval_time']:f})")
        # distillation
        else:
            # Load pretrained teacher model (use the same tokenizer as student)
            teacher_config = AutoConfig.from_pretrained(
                args.config_name if args.config_name else args.teacher_model_name_or_path,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )

            teacher_model = AutoModelForQuestionAnswering.from_pretrained(
                args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in args.teacher_model_name_or_path),
                config=teacher_config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )

            teacher_model.to(args.device)

            global_step, tr_loss = distill(
                args,
                train_dataset,
                teacher_model,
                model,
                tokenizer)

            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            


    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        output_dir = os.path.join(args.output_dir, "checkpoint-last")
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and not args.do_search and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [os.path.join(args.output_dir, "checkpoint-best")]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1]  # if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, config=config)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result, evalTime = evaluate(args, model, tokenizer, prefix=prefix)
            dump_metrics(result, evalTime, args)

            result = dict((f"{k}_{global_step}", v) for k, v in result.items())
            results.update(result)

        logger.info("Results: {}".format(results))

    # Evolutionary Search
    if args.do_search and args.local_rank in [-1, 0]:
        import warnings
        warnings.filterwarnings("ignore")

        # assert args.population_size == args.parent_size + args.mutation_size + args.crossover_size
        evolution = Evolution(model, args, evaluate, tokenizer)
        evolution.load_store(os.path.join(args.model_name_or_path, 'store.tsv'))

        lower_gene = sample_length_configuration(
            args.max_seq_length,
            config.num_hidden_layers,
            length_drop_ratio=args.length_drop_ratio_bound,
        )
        upper_gene = (args.max_seq_length,) * config.num_hidden_layers
        evolution.add_gene(lower_gene, method=0)
        evolution.add_gene(upper_gene, method=0)
        evolution.lower_constraint = evolution.store[lower_gene][0]
        evolution.upper_constraint = evolution.store[upper_gene][0]

        length_drop_ratios = [inverse(r) for r in np.linspace(approx_ratio(args.length_drop_ratio_bound), 1, args.population_size + 2)[1:-1]]
        for p in length_drop_ratios:
            gene = sample_length_configuration(
                args.max_seq_length,
                config.num_hidden_layers,
                length_drop_ratio=p,
            )
            evolution.add_gene(gene, method=0)

        for i in range(args.evo_iter + 1):
            logger.info(f"| Start Iteration {i}:")
            population, area = evolution.pareto_frontier()
            parents = evolution.convex_hull()
            results = {"area": area, "population_size": len(population), "num_parents": len(parents)}

            logger.info(f"| >>>>>>>> {' | '.join([f'{k} {v}' for k, v in results.items()])}")
            for gene in parents:  # population
                logger.info("| " + store2str(gene, *evolution.store[gene][:3]))

            evolution.save_store(os.path.join(args.output_dir, f'store-iter{i}.tsv'))
            evolution.save_population(os.path.join(args.output_dir, f'population-iter{i}.tsv'), population)
            evolution.save_population(os.path.join(args.output_dir, f'parents-iter{i}.tsv'), parents)

            if i == args.evo_iter:
                break

            k = 0
            while k < args.mutation_size:
                if evolution.mutate(args.mutation_prob):
                    k += 1

            k = 0
            while k < args.crossover_size:
                if evolution.crossover():
                    k += 1

    # HPO Search
    if args.do_optuna_search and args.local_rank in [-1, 0]:
        import optuna
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
        size = (1, args.max_seq_length)
        dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.zeros(size, dtype=torch.long).to(args.device),
        )
        if hasattr(model, "module"):
            model.module.config.output_attentions = True
        else:
            model.config.output_attentions = True

        def objective(trial):
            # 384 - 307 - 246 - 197 - 157 - 126 - 101 - 81
            # [384, 277] [277, 222] [ 222, 177] [177, 142] [ 142, 114] [114, 91]
            hlayer_0 = trial.suggest_int("hlayer_0", high=384, low=91)
            hlayer_1 = trial.suggest_int("hlayer_1", high=hlayer_0, low=91)
            hlayer_2 = trial.suggest_int("hlayer_2", high=hlayer_1, low=91)
            hlayer_3 = trial.suggest_int("hlayer_3", high=hlayer_2, low=91)
            hlayer_4 = trial.suggest_int("hlayer_4", high=hlayer_3, low=91)
            hlayer_5 = trial.suggest_int("hlayer_5", high=hlayer_4, low=91)
            model.eval()
            if model.config.model_type == "distilbert":
                bert = model.distilbert
            else:
                assert hasattr(model, "bert")
                bert = model.bert
            bert.set_length_config((hlayer_0, hlayer_1, hlayer_2, hlayer_3, hlayer_4, hlayer_5))
            macs = torchprofile.profile_macs(model, args=dummy_inputs)
            results, evalTime = evaluate(args, model, tokenizer, dataset=dataset, examples=examples, features=features)
            dump_metrics(results, evalTime, args, macs=macs, rate=-1.)

            return results['best_f1'], evalTime

        # TPESampler covers MOTPESampler starting from 2.9.0
        # sampler = optuna.samplers.MOTPESampler(n_startup_trials=50, n_ehvi_candidates=24, seed=12)
        # default TPE option gets better pareto-front instead of multivariate=True, group=True
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name='Dynamic-TinyBERT', directions=['maximize', 'minimize'], sampler=sampler,
                                    storage='sqlite:///' + 'Dynamic-TinyBERT', load_if_exists=True)
        study.optimize(objective, n_trials=150)

        for trial in study.best_trials:
            res = trial.params
            res['f1'] = trial.values[0]
            res['evalTime'] = trial.values[1]
            with open('optuna-pareto-front.log', 'a') as f:
                json.dump(res, f)
                f.write('\n')

    # HPO Search
    if args.do_sigopt_search and args.local_rank in [-1, 0]:
        from sigopt import Connection
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
        size = (1, args.max_seq_length)
        dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.ones(size, dtype=torch.long).to(args.device),
            torch.zeros(size, dtype=torch.long).to(args.device),
        )
        if hasattr(model, "module"):
            model.module.config.output_attentions = True
        else:
            model.config.output_attentions = True

        def objective(assignments):
            # 384 - 307 - 246 - 197 - 157 - 126 - 101 - 81
            # [384, 277] [277, 222] [ 222, 177] [177, 142] [ 142, 114] [114, 91]
            hpo_length_config = [round(assignments[f'x{i}']) for i in range(6)]

            model.eval()
            if model.config.model_type == "distilbert":
                bert = model.distilbert
            else:
                assert hasattr(model, "bert")
                bert = model.bert
            bert.set_length_config(hpo_length_config)
            macs = torchprofile.profile_macs(model, args=dummy_inputs)
            results, evalTime = evaluate(args, model, tokenizer, dataset=dataset, examples=examples, features=features)
            dump_metrics(results, evalTime, args, macs=macs, rate=-1.)

            return [{'name': 'f1', 'value': results['best_f1']}, {'name': 'evalTime', 'value': evalTime}]

        conn = Connection()
        experiment_meta = {
            'name': 'Dynamic-TinyBERT',
            #'project': 'hw-knobs',
            # constaints only support double for now
            'parameters': [{'name': f'x{k}', 'type': 'double', 'bounds': {'min': 91, 'max': 384}} for k in range(6)],
            'metrics': [{'name': 'f1', 'strategy': 'optimize', 'objective': 'maximize'},
                        {'name': 'evalTime', 'strategy': 'optimize', 'objective': 'minimize'}],
            'linear_constraints': [{'type': 'less_than', 'threshold': 0,
                                    'terms': [{'name': f'x{k}', 'weight': -1}, {'name': f'x{k + 1}', 'weight': 1}]}
                                   for k in range(5)],
            'observation_budget': 150,
            'parallel_bandwidth': 1
        }
        experiment = conn.experiments().create(**experiment_meta)
        logger.info("created experiment: https://app.sigopt.com/experiment/" + experiment.id)

        # run with the budget
        while experiment.progress.observation_count < experiment.observation_budget:
            suggestion = conn.experiments(experiment.id).suggestions().create()
            values = objective(suggestion.assignments)
            obs = conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id, values=values)
            experiment = conn.experiments(experiment.id).fetch()

        # Fetch the best configuration and explore your experiment
        all_best_assignments = conn.experiments(experiment.id).best_assignments().fetch()
        for data in all_best_assignments.data:
            res = data.assignments
            for m in data.values:
                res[m.name] = m.value
            with open('sigopt-pareto-front.log', 'a') as f:
                json.dump(res, f)
                f.write('\n')


if __name__ == "__main__":
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"huggingface/transformers version: {transformers.__version__}")
    main()
