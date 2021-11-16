# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base pruning scheduler class
"""

import abc

from ...utils import Config
from .schedulers_utils import parse_model_for_pruning


class PruningScheduler(abc.ABC):
    """Manages and schedules pruning of a module and all consecutive modules"""

    SCHEDULER_CONFIG = None
    PRUNING_FN_DICT = {}

    def __init__(self, model, config, pruning_fn=None, tb_writer=None, initial_step=0):
        """Init pruning scheduler, with model to be pruned and config for more specific pruning options (pruning per layer parameters)"""
        if type(config) != self.SCHEDULER_CONFIG:
            raise TypeError(
                f"Expected scheduler config from type {self.SCHEDULER_CONFIG.__name__}, got type {type(config).__name__}")
        super().__init__()
        self.model = model
        self.config = config
        self.writer = tb_writer
        self.pruning_fn = pruning_fn
        if not callable(self.pruning_fn):
            if self.pruning_fn is None:
                try:
                    self.pruning_fn = config.pruning_fn
                except AttributeError:
                    raise AttributeError(
                        "No pruning function was supplied to the scheduler.\npruning_fn is None and no pruning_fn entry in config")
            try:
                self.pruning_fn = self.PRUNING_FN_DICT[self.pruning_fn]
            except KeyError:
                raise NotImplementedError(f"pruning fucntion '{pruning_fn}' is not supported.\n"
                                          f"Accepted pruning functions: {self.PRUNING_FN_DICT.keys()}")
        self.method_dicts = []
        self.init_prune()
        self.global_step = initial_step

    @abc.abstractmethod
    def step(self):
        """perform a step of the scheduler according to the scheduler policy"""
        self.global_step += 1

    def remove_pruning(self):
        """Remove existing pruning methods from model"""
        for method_dict in self.method_dicts:
            method_dict['method'].remove()

    def tb_log(self):
        """Log scheduler information to tensorboard"""
        if self.writer is not None:
            # log sparsity per layer and average sparsity of pruned layers
            total_nonzero = 0
            total_params = 0
            for method_dict in self.method_dicts:
                method = method_dict['method']
                assert hasattr(method.module, method.name)
                pruned_tensor = getattr(method.module, method.name)
                nonzero = pruned_tensor.count_nonzero()
                params = pruned_tensor.numel()
                self.writer.add_scalar(
                    'sparsity_per_layer/' + method_dict['name'], 1 - nonzero / params, self.global_step)
                total_nonzero += nonzero
                total_params += params
            if total_params != 0:
                self.writer.add_scalar('pruning/average_model_sparsity',
                                       1 - total_nonzero / total_params, self.global_step)

    def extra_repr(self):
        return 'tb_writer={},\nconfig={}'.format(self.writer is not None, self.config)

    def __repr__(self):
        extra = '\n  '.join(str(self.extra_repr()).split('\n')).strip()
        s = "{}(\n  {}\n)".format(
            type(self).__name__,
            extra
        )
        return s

    def print_pruning_methods(self, flush=False):
        """Print pruning methods initialized by the scheduler"""
        print(f"{type(self).__name__} Pruned Methods:", flush=flush)
        for md in self.method_dicts:
            print(md, flush=flush)

    def init_prune(self):
        """Initialize pruning methods in pruned model"""
        parsed_model = parse_model_for_pruning(self.model, self.config)
        for name, module in self.model.named_modules():
            if name in parsed_model:
                method_dict = {'name': name, 'kwargs': parsed_model[name]}
                method_dict['method'] = self.pruning_fn(
                    module, **parsed_model[name])
                self.method_dicts.append(method_dict)


class PruningConfig(Config):
    """
    Pruning scheduler configuration object

    Arguments: 
        pruning_fn (str, func): (default: None)
        pruning_fn_default_kwargs (dict, optional): default arguments that will be used when 
            calling pruning_fn (default: {})
        prune_layer_types (dict(str: kwargs), optional): keys are layers type __name__ attribute
            that should be pruned and kwargs is a dict containing the attributes to be passed to
            pruning_fn. Attributes here will override the default attributes in pruning_fn_default_kwargs
            (default: 
                {
                    "Linear": {"name": "weight"},
                    "Conv2d": {"name": "weight"}
                }
            )
        not_to_prune (list(str), optional): layers which contains any of the strings in
            the `not_to_prune` list will not be pruned. Regex may be used in this field.
            overrides `prune_layer_types` and `weight_sparsity_map` (default: [])
        weight_sparsity_map (dict(str: kwargs), optional): overrides default and layer types kwargs
            for specific weights defined in `prune_layer_type` dict, overrided by `not_to_prune` list. 
            Regex may be used here. E.g. {"m1": {"target_sparsity": 0.5}} (default: {})
        explicit_prune (dict(str: kwargs), optional): weights to be explicitly pruned, 
            overrides `not_to_prune`. E.g. {"c_module1": {"name": "c_weight", "target_sparsity": 0.5}}
            (default: {})
        init_strategy: initialization strategy for pruning. (default: "uniform")
        initial_step: Start scheduler with global_step=initial_step
    """
    ATTRIBUTES = Config.add_attributes(
        {
            "pruning_fn": None,
            "pruning_fn_default_kwargs": {},
            "prune_layer_types": {
                "Linear": {"name": "weight", },
                "Conv2d": {"name": "weight", },
            },
            "not_to_prune": [],
            "weight_sparsity_map": {},
            "explicit_prune": {},
            "init_strategy": "uniform",
            "initial_step": 0,
            "scheduler": None,
        }
    )
