# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import re
import weakref
from functools import wraps
from abc import abstractmethod, ABC

import torch
from torch import nn

from . import utils


def _freeze(module, name):
    """
    Freeze parameter in module by converting to it to a buffer.
    Output: True if succeeded
    """
    p = getattr(module, name, None)
    if type(p) is nn.Parameter:
        delattr(module, name)
        module.register_buffer(name, p.data)
        return True
    return False


def _unfreeze(module, name):
    """
    Unfreezes a tensor in module by converting it to a a parameter
    Output: True if succeeded or named tensor is already a parameter
    """
    p = getattr(module, name, None)
    if p is None or not isinstance(p, torch.Tensor):
        return False
    if type(p) is not nn.Parameter:
        delattr(module, name)
        module.register_parameter(name, nn.Parameter(p))
    return True


def freeze_parameters(module, *names, excluding=False):
    frozen_names = []
    for n, _ in list(module.named_parameters(recurse=False)):
        # Either excluding and n not in names or not excluding in n in names
        if excluding ^ (n in names):
            if _freeze(module, n):
                frozen_names.append(n)
    return frozen_names


def unfreeze_parmeters(module, *names):
    unfrozen_names = []
    for name in names:
        if _unfreeze(module, name):
            unfrozen_names.append(name)
    return unfrozen_names


class _AddOn(nn.Module, ABC):
    ADDON_NAME = ''
    CONFIG = None

    def __init__(self, module, config=None):
        super().__init__()
        if hasattr(module, self.ADDON_NAME):
            raise RuntimeError(f"Module already has {self.ADDON_NAME} addon")
        setattr(module, self.ADDON_NAME, self)
        self._module = weakref.ref(module)
        self.module_forward = module.forward
        forward = wraps(module.forward.__func__)(self.get_modified_forward())
        module.forward = forward.__get__(module)
        self.frozen_parameters = []
        self.config = config
        if not isinstance(self.config, self.CONFIG):
            raise RuntimeError(
                f"Wrong config instance, expected {self.CONFIG.__name__}, got {self.config.__class__.__name__}.")

    def freeze_parameters(self, *names, excluding=False):
        self.frozen_parameters += freeze_parameters(
            self.module, *names, excluding=excluding)

    def unfreeze_parameters(self):
        """Unfreezes all previously frozen parameters"""
        unfreeze_parmeters(self.module, *self.frozen_parameters)

    @classmethod
    def _mapping_name(cls):
        return cls.ADDON_NAME + '_MAPPING'

    @classmethod
    def _addon_mapping(cls):
        try:
            out = getattr(cls, cls._mapping_name())
        except AttributeError:
            out = {}
            setattr(cls, cls._mapping_name(), out)
        return out

    @classmethod
    def register_addon(cls, module_cls):
        def register(addon):
            cls._addon_mapping()[module_cls] = addon
            return addon
        return register

    @property
    def module(self):
        return self._module()

    @classmethod
    def apply_to(cls, module, config):
        cls._addon_mapping()[type(module)](module, config)

    @classmethod
    def apply_to_model(cls, model, config):
        """
        Method to convert model to PEFT method for fine-tuning
        """
        for name, m in list(model.named_modules()):
            module_config = [v for k, v in config.layers.items(
            ) if re.search(k, name) is not None]
            if len(module_config) == 1:
                cls.apply_to(m, module_config[0] if isinstance(
                    module_config[0], cls.CONFIG) else cls.CONFIG.from_dict(module_config[0]))
            elif len(module_config) > 1:
                raise RuntimeError("More than 1 match in config")
            for n, _ in list(m.named_parameters(recurse=False)):
                if not any([re.search(p, '.'.join([name, n])) for p in config.exceptions + [cls.ADDON_NAME]]):
                    freeze_parameters(m, n)

    @abstractmethod
    def get_modified_forward(self):
        pass

    def forward(self):
        raise RuntimeError()

    @classmethod
    def get_from(cls, module):
        return getattr(module, cls.ADDON_NAME, None)

    @classmethod
    def get_modified_modules(cls, model):
        for m in model.modules():
            if cls.get_from(m) is not None:
                yield m

    def embed_remove(self, *, unfreeze=False):
        self.embed()
        if unfreeze:
            self.unfreeze_parameters()
        self.module.forward = self.module_forward
        delattr(self.module, self.ADDON_NAME)

    @classmethod
    def embed_remove_from(cls, module, *, unfreeze=False):
        cls.get_from(module).embed_remove(unfreeze=unfreeze)

    # TODO: handle unfreezing
    @classmethod
    def embed_remove_from_model(cls, model):
        for m in list(cls.get_modified_modules(model)):
            cls.embed_remove_from(m)

    @abstractmethod
    def embed(self):
        """
        Embed AddOn to its module
        """


class _AddOnModuleConfig(utils.Config):
    """Base class for _AddOn module config"""


class IA3ModuleConfig(_AddOnModuleConfig):
    ATTRIBUTES = {
        "pre_scale": False,
        "post_scale": False,
    }


class _AddOnModelConfig(utils.Config):
    """Base class for _AddOn model config"""
    ATTRIBUTES = {
        "layers": {},
        "exceptions": [],
    }


class IA3ModelConfig(_AddOnModelConfig):
    pass


class IA3AddOn(_AddOn):
    ADDON_NAME = "ia3"
    CONFIG = IA3ModuleConfig

    @classmethod
    def apply_to_model(cls, model, config=None):
        """
        Apply IA^3 as described in [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning]
        (https://arxiv.org/abs/2205.05638)
        """
        # BERT default config
        if config is None:
            d = IA3ModuleConfig(
                pre_scale=True,
                post_scale=True
            )
            config = IA3ModelConfig(
                layers={
                    "key": d,
                    "value": d,
                    "query": d,
                    r"dense": d,
                    # r"\d+\.output\.dense": d,
                },
                exceptions=['classifier', 'pooler',
                            'qa_outputs', 'teacher', 'cls']
            )
        super().apply_to_model(model, config)


@IA3AddOn.register_addon(nn.Linear)
class IA3LinearAddOn(IA3AddOn):
    def __init__(self, module, config):
        assert type(
            module) is nn.Linear, f"{self.__class__.__name__} expected module of type nn.Linear, got {module.__class__.__name__}"
        super().__init__(module, config)
        self.pre_scale = None
        self.post_scale = None
        if config.pre_scale:
            self.pre_scale = nn.Parameter(torch.ones(module.in_features))
        if config.post_scale:
            self.post_scale = nn.Parameter(torch.ones(module.out_features))

    def get_modified_forward(self):
        def forward(module, input):
            if self.pre_scale is not None:
                input = input * self.pre_scale
            input = self.module_forward(input)
            if self.post_scale is not None:
                input = input * self.post_scale
            return input
        return forward

    @torch.no_grad()
    def embed(self):
        if self.pre_scale is not None:
            self.module.weight *= self.pre_scale
        if self.post_scale is not None:
            self.module.weight *= self.post_scale.view(-1, 1)
            if self.module.bias is not None:
                self.module.bias *= self.post_scale


class LoraModuleConfig(_AddOnModuleConfig):
    ATTRIBUTES = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.,
    }


class LoraModelConfig(_AddOnModelConfig):
    pass


class LoraAddOn(_AddOn):
    ADDON_NAME = "lora"
    CONFIG = LoraModuleConfig

    @classmethod
    def apply_to_model(cls, model, config=None):
        """
        Apply LoRA as described in [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS]
        (https://arxiv.org/abs/2106.09685)
        """
        # BERT default config
        if config is None:
            d = LoraModuleConfig(
                rank=8,
                alpha=16,
                dropout=0.,
            )
            config = LoraModelConfig(
                layers={
                    "value": d,
                    "query": d,
                },
                exceptions=['classifier', 'pooler',
                            'qa_outputs', 'teacher', 'cls']
            )
        super().apply_to_model(model, config)

    def extra_repr(self):
        s = super().extra_repr()
        if self.rank > 0:
            s += f"rank={self.rank}, alpha={self.rank * self.scaling}"
        return s


@LoraAddOn.register_addon(nn.Linear)
class LoraLinearAddOn(LoraAddOn):
    def __init__(self, module, config):
        assert type(
            module) is nn.Linear, f"{self.__class__.__name__} expected module of type nn.Linear, got f{module.__class__.__name__}"
        super().__init__(module, config)
        self.rank = config.rank
        self.dropout = nn.Dropout(config.dropout)
        if self.rank > 0:
            self.scaling = config.alpha / config.rank
            self.A = nn.Parameter(torch.zeros(
                self.rank, self.module.in_features))
            self.B = nn.Parameter(torch.zeros(
                self.module.out_features, self.rank))
            nn.init.kaiming_normal_(self.A, a=math.sqrt(5))

    def get_modified_forward(self):
        def forward(module, input):
            output = self.module_forward(input)
            lora_out = 0.
            if self.rank > 0:
                lora_out = self.dropout(
                    input) @ self.A.T @ self.B.T * self.scaling
            return output + lora_out
        return forward

    @torch.no_grad()
    def embed(self):
        if self.rank > 0:
            self.module.weight += (self.A.T @ self.B.T).T * self.scaling
    
    def extra_repr(self):
        s = super().extra_repr()
        if self.rank > 0:
            s += f"rank={self.rank}, alpha={self.rank * self.scaling}"
        return s


def bitfit_apply(model):
    """
    Apply BitFit as described in [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models]
    (https://arxiv.org/abs/2106.10199)
    """
    bitfit_exceptions = ['classifier', 'pooler',
                         'qa_outputs', 'teacher', 'cls']
    for n, m in model.named_modules():
        exception = any([e in n for e in bitfit_exceptions])
        if type(m) in (nn.Embedding, nn.Linear, nn.LayerNorm) and not exception:
            weight = m.weight.detach().clone()
            del m.weight
            m.register_buffer('weight', weight)
