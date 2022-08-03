# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
API utilities for using model compression package
"""
import json
import logging
from functools import wraps
from argparse import Action
import inspect

import torch
from torch import nn
from torch.nn import functional as F

from .pruning.schedulers import *
from . import distiller
from .pruning import registry
from .quantization import Quantizer, QuantizerConfig


logger = logging.getLogger(__name__)


# Pruning API utilities
def remove_pruning(module, name='weight'):
    """Embed mask to the pruned weight and remove the pruning method"""
    module.get_pruning_parameters('method', name=name).remove()
    return module


def get_tensor_sparsity_ratio(tensor):
    """Compute sparsity ratio of tensor given"""
    tensor = tensor.detach()
    return (1 - tensor.count_nonzero() / tensor.numel()).item()


class ConcatenateStringAction(Action):
    """argparse Action to concatenate arguments into a single string with space seperator"""

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, ' '.join(values))


def add_pruning_arguments_to_parser(parser):
    """Add pruning arguments to existing argparse parser"""
    parser.add_argument('--do_prune', action='store_true',
                        help="Perform pruning when training a model")
    parser.add_argument('--pruning_config', type=str,
                        default='', help="Path to a pruning config")
    parser.add_argument('--pruning_override', type=str, nargs='*', action=ConcatenateStringAction,
                        default='', help="JSON string to override pruning configuration file")
    return parser


def pruning_config_factory(config, *args, **kwargs):
    """Construct a pruning config from file and override it with args and kwargs"""
    if isinstance(config, PruningConfig):
        return config
    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)
    config_type = config['scheduler']
    c = registry.get_config_class(config_type).from_dict(config)
    c.update(*args, **kwargs)
    return c


def pruning_scheduler_factory(model, config, *args, **kwargs):
    """Construct a pruning scheduler from config, config can be overrided with args and kwargs"""
    config = pruning_config_factory(config, *args, **kwargs)
    return registry.get_scheduler_class(config.scheduler)(model, config)


def get_linear_rewinding_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, prune_start_step, prune_end_step, rewind_interval, last_epoch=-1):
    """Return linear decay schedule with warmup with learning rate rewinding"""
    def pruning_lr_lambda(current_step: int):
        prune_interval = prune_end_step - prune_start_step
        train_wo_pruning_steps = num_training_steps - prune_interval
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1., num_warmup_steps))
        if current_step >= prune_start_step and current_step < prune_end_step:
            rewind = (current_step -
                      prune_start_step) // rewind_interval * rewind_interval
            return max(0., float(train_wo_pruning_steps - current_step + rewind) / float(max(1, train_wo_pruning_steps - num_warmup_steps)))
        b = prune_interval if current_step < prune_start_step else 0.
        return max(0., float(num_training_steps - b - current_step) / float(max(1, train_wo_pruning_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, pruning_lr_lambda, last_epoch)


# Quantization API utilities
def quantized_model_class_factory(cls, config):
    """Construct a qunatized model class from existing model class and cofig"""
    class QuantizedModelClass(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            Quantizer(self, config).quantize()
    QuantizedModelClass.__name__ = 'Quantized' + cls.__name__
    return QuantizedModelClass


def add_quantization_arguments_to_parser(parser):
    """Add quantization arguments to existing argparse parser"""
    parser.add_argument("--do_quantization", action="store_true",
                        help="Perform quantization aware training")
    parser.add_argument("--quantization_config", type=str,
                        default=None, help="Path to quantization config")
    parser.add_argument("--quantization_override", type=str, nargs='*', default='',
                        help="Override attributes in loaded quantization config. Format: key:value delimited by spaces. E.g.: key1:value1 key2:value21,value22")


def convert_model_for_qat(model, config):
    """Convert model for quantization aware training according to given config"""
    return Quantizer(model, config).quantize()


def quantization_config_factory(args):
    """Construct a quantization config from file and override it with args and kwargs"""
    if args.quantization_config:
        config = QuantizerConfig.from_json_file(args.quantization_config)
    else:
        config = QuantizerConfig()
    override_args = {}
    for k_v in args.quantization_override:
        k, v = k_v.split(':')
        override_args[k] = v if len(v.split(',')) == 1 else v.split(',')
    config.update_from_dict(override_args)
    return config


def quantization_model_or_class_factory(args, *, model=None, cls=None):
    """Return either converted model for quantization or a quantized version of the class passed"""
    if model is not None and cls is not None:
        raise AttributeError(
            "Pass either model or class to be converted for quantization")
    if args.do_quantization:
        config = quantization_config_factory(args)
        logger.info(f"Quantizing model with config: {config}")
        if model is not None:
            return convert_model_for_qat(model, config)
        if cls is not None:
            return quantized_model_class_factory(cls, config)
    return model if model is not None else cls


# Pruning integration into HF/transformers
try:
    import os

    import torch
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            SummaryWriter = lambda *args, **kwargs: None

    from transformers import TrainerCallback
    from transformers import logging as hf_logging

    logger = hf_logging.get_logger(__name__)

    class HFTrainerPruningCallback(TrainerCallback):
        """Callback integrating pruning to transformers.Trainer"""

        def __init__(self, pruning_args):
            self.args = pruning_args
            if self.args.do_prune:
                self.config = pruning_config_factory(
                    self.args.pruning_config, self.args.pruning_override)
            self.scheduler = None

        def on_init_end(self, args, state, control, model=None, **kwargs):
            # Init pruning only if model is going to train
            if self.args.do_prune and args.do_train:
                assert kwargs.pop(
                    'optimizer') is None, "Optimizer needs to be initialized after pruning initialization"
                self.scheduler = pruning_scheduler_factory(model, self.config)
                if args.local_rank in [-1, 0]:
                    print("Applying Pruning: {}".format(self.scheduler))
                    self.scheduler.print_pruning_methods(flush=True)
                self.scheduler.writer = SummaryWriter(args.logging_dir)

        def on_step_end(self, args, state, control, **kwargs):
            if self.args.do_prune:
                self.scheduler.step()

        def on_log(self, args, state, control, **kwargs):
            if self.args.do_prune:
                self.scheduler.tb_log()

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if self.args.do_prune:
                if self.scheduler.writer is not None:
                    self.scheduler.writer.close()
                state_dict = model.module.state_dict() if hasattr(
                    model, 'module') else model.state_dict()
                if model._keys_to_ignore_on_save is not None:
                    state_dict = {k: v for k, v in state_dict.items(
                    ) if k not in self._keys_to_ignore_on_save}
                torch.save(state_dict, os.path.join(
                    args.output_dir, 'raw_pytorch_model.bin'))
                self.scheduler.remove_pruning()
                with open(os.path.join(args.output_dir, 'pruning_config.json'), 'w') as file:
                    file.write(self.config.to_json_string())

    class HFTeacherWrapper(distiller.TeacherWrapper):
        """Distillation teacher wrapper adjusted to work with HuggingFace/transformers"""

        def __init__(self, *args, **kwargs):
            dtype = next(args[0].parameters()).dtype
            logit_names = kwargs.pop('logit_names')
            self.weight = kwargs.pop('ce_weight', None)
            self.hidden_alpha = kwargs.pop('hidden_alpha', None)
            self.attention_alpha = kwargs.pop('attention_alpha', None)
            similarity_loss = kwargs.pop('similarity_loss', 'mse')
            super().__init__(*args, **kwargs)
            self.logit_names = logit_names if isinstance(
                logit_names, list) else [logit_names]
            if self.weight is None:
                self.weight = torch.ones(
                    len(self.logit_names)) / len(self.logit_names)
            self._input = None
            # Teacher hack, model will look for a tensor to determine dtype because
            # there are no parameters in the model. So we add a single parameter
            # to help the model determine dtype
            for m in self.teacher.children():
                m.dtype_parameter_hack = nn.Parameter(
                    torch.tensor([1], dtype=dtype))
            if self.hidden_alpha is not None:
                def similarity_loss_fn_factory():
                    similarity_loss_fn = lambda *args: getattr(
                        F, similarity_loss + '_loss')(*args, reduction='mean')
                    if similarity_loss == 'cosine_embedding':
                        def ret(x, y): return similarity_loss_fn(
                            x, y, torch.ones(1, device=x.device))
                    else:
                        ret = similarity_loss_fn
                    return ret
                self.similarity_loss_fn = similarity_loss_fn_factory()

        def forward(self, **inputs):
            # Delete _input to maek room for new input in case it wasn't deleted already
            if hasattr(self, '_input'):
                del self._input
            teacher_inputs_ = inputs.copy()
            # Delete labels from teacher input to get logits instead of computed loss
            if 'labels' in teacher_inputs_:
                del teacher_inputs_['labels']
            assert self.hidden_alpha is None or teacher_inputs_.get(
                'output_hidden_states', False), "Cross model hidden states distillation loss is used but model doesn't output hidden_states"
            assert self.attention_alpha is None or teacher_inputs_.get(
                'output_attentions', False), "Cross model attention states distillation loss is used but model doesn't output attentions"
            teacher_inputs = {}
            for k, v in teacher_inputs_.items():
                if torch.is_tensor(v):
                    v = v.detach().clone()
                teacher_inputs[k] = v
            self._input = inputs
            return super().forward(**teacher_inputs)

        def _masked_outputs(self, logit, size):
            """Mask outputs of padding tokens"""
            mask = self._input["attention_mask"].unsqueeze(-1).bool()
            return torch.masked_select(logit, mask).view(-1, size)

        def compute_distill_loss_callback(self, student_outputs, teacher_outputs):
            """
            Compute distillation loss of student w.r.t teacher outputs.
            Callback is overrided to support DistilBERT and TinyBERT like distillation
            """
            loss = 0
            if self.ce_alpha != 0:
                ce_loss = 0
                for logit_name, weight in zip(self.logit_names, self.weight):
                    student_logit = student_outputs[logit_name]
                    teacher_logit = teacher_outputs[logit_name]
                    # preprocess LM logits, we consider each token as an example and average the loss over all tokens
                    vocab_size = self.teacher.config.vocab_size
                    if len(teacher_logit.size()) > 2 and teacher_logit.size()[-1] == vocab_size:
                        teacher_logit = self._masked_outputs(
                            teacher_logit, vocab_size)
                        student_logit = self._masked_outputs(
                            student_logit, vocab_size)
                        assert teacher_logit.size() == student_logit.size()
                    ce_loss += weight * \
                        self.compute_cross_entropy_loss(
                            student_logit, teacher_logit)
                loss += ce_loss * self.ce_alpha
            # From distilbert and tinybert. Add loss term with cosine similarity of the embedding output of BERT
            for key, alpha_dict in zip(['hidden_states', 'attentions'], [self.hidden_alpha, self.attention_alpha]):
                if alpha_dict is not None:
                    cosine_loss = 0
                    for i in range(len(student_outputs[key])):
                        if alpha_dict[i] != 0:
                            teacher_state = teacher_outputs[key][i]
                            student_state = student_outputs[key][i]
                            assert teacher_state.size() == student_state.size()
                            hidden_size = teacher_state.size(-1)
                            if key == 'hidden_states':
                                teacher_state = self._masked_outputs(
                                    teacher_state, hidden_size)
                                student_state = self._masked_outputs(
                                    student_state, hidden_size)
                            unscaled_loss = self.similarity_loss_fn(
                                student_state, teacher_state)
                            cosine_loss += alpha_dict[i] * unscaled_loss
                    loss += cosine_loss
            return loss

        def compute_distill_loss(self, student_outputs, teacher_outputs=None):
            loss = super().compute_distill_loss(
                student_outputs, teacher_outputs=teacher_outputs)
            # Delete _input to conserve memory after distillation loss calculation is over
            del self._input
            return loss

    def hf_add_teacher_to_student(
        student,
        teacher,
        *,
        student_alpha=0.5,
        teacher_ce_alpha=0.5,
        teacher_hidden_alpha=None,
        teacher_attention_alpha=None,
        teacher_ce_temperature=1.,
        teacher_similarity_loss='mse',
        teacher_logit_names='logits',
        teacher_ce_weights=None,
        teacher_convert_parameters=True,
    ):
        """Add model distillation from teacher to HuggingFace/transformers model"""
        teacher_sig = inspect.signature(teacher.forward)
        student_sig = inspect.signature(student.forward)
        teacher_unique = set(teacher_sig.parameters) - \
            set(student_sig.parameters)
        student_unique = set(student_sig.parameters) - \
            set(teacher_sig.parameters)

        teacher = HFTeacherWrapper(
            teacher,
            ce_alpha=teacher_ce_alpha,
            hidden_alpha=teacher_hidden_alpha,
            attention_alpha=teacher_attention_alpha,
            ce_temperature=teacher_ce_temperature,
            logit_names=teacher_logit_names,
            ce_weight=teacher_ce_weights,
            convert_parameters=teacher_convert_parameters,
            similarity_loss=teacher_similarity_loss,
        )
        student._teacher = teacher
        student._forward = student.forward

        @wraps(student.forward.__func__)
        def forward(student, *args, **kwargs):
            if student.training:
                if teacher_hidden_alpha is not None:
                    kwargs.update(output_hidden_states=True)
                if teacher_attention_alpha is not None:
                    kwargs.update(output_attentions=True)
            student_kwargs = {k: kwargs[k]
                              for k in kwargs if k not in teacher_unique}
            student_output = student._forward(*args, **student_kwargs)
            if student.training:
                teacher_kwargs = {k: kwargs[k]
                                  for k in kwargs if k not in student_unique}
                student._teacher(*args, **teacher_kwargs)
                loss = student_output["loss"] * student_alpha
                loss += student._teacher.compute_distill_loss(
                    student_output)
                student_output["loss"] = loss
            return student_output

        new_sig = student_sig.replace(parameters=list(inspect.signature(
            student.forward.__func__).parameters.values()) + [teacher_sig.parameters[k] for k in teacher_unique])
        forward.__signature__ = new_sig

        student.forward = forward.__get__(student)
        return student

    def hf_remove_teacher_from_student(student):
        """Remove model distillation and teacher from HuggingFace/transformers model"""
        student.forward = student._forward
        del student._teacher
        del student._forward
        return student

except ModuleNotFoundError:
    def raise_hf_import_error(name):
        raise ImportError(
            f"{name} requires HuggingFace/transformers library. Install with `pip install transformers`")

    class HFTrainerPruningCallback:
        def __init__(self, *args, **kwargs):
            raise_hf_import_error(self.__class__.__name__)

    class HFTeacherWrapper:
        def __init__(self, *args, **kwargs):
            raise_hf_import_error(self.__class__.__name__)

    def hf_add_teacher_to_student(*args, **kwargs):
        raise_hf_import_error(hf_add_teacher_to_student.__name__)

    def hf_remove_teacher_from_student(*args, **kwargs):
        raise_hf_import_error(hf_remove_teacher_from_student.__name__)
