# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utils for schedulers
"""
import logging
import re

import numpy as np


logger = logging.getLogger(__name__)


def parse_explicit_prune(model, config):
    """Returns a dictionary of names of modules and their pruning parameters"""
    parsed_explicit_prune = {}
    for raw, explicit_kwargs in config.explicit_prune.items():
        pat = re.compile(raw)
        for name, _ in filter(lambda t: pat.search(t[0]) and hasattr(t[1], explicit_kwargs["name"]), model.named_modules()):
            kwargs = {}
            kwargs.update(config.pruning_fn_default_kwargs)
            kwargs.update(explicit_kwargs if explicit_kwargs else {})
            parsed_explicit_prune[name] = kwargs
    return parsed_explicit_prune


def parse_not_to_prune(model, config):
    """Returns a list of names of modules not to prune in the model"""
    patterns = [re.compile(s) for s in config.not_to_prune]
    parsed_not_to_prune = []
    for name, module in model.named_modules():
        if type(module).__name__ in config.prune_layer_types:
            if any([p.search(name) for p in patterns]):
                parsed_not_to_prune.append(name)
    return parsed_not_to_prune


def parse_sparsity_map(model, config):
    """Returns a dictionary of names of modules and their pruning parameters"""
    parsed_sparsity_map = {}
    for name, module in model.named_modules():
        if type(module).__name__ in config.prune_layer_types:
            sparsity_map = [config.weight_sparsity_map[s]
                            for s in config.weight_sparsity_map if re.search(s, name)]
            if len(sparsity_map) > 1:
                raise RuntimeError(
                    f"There are more than two entries in weight_sparsity_map that match parameter: {name}")
            kwargs = {}
            kwargs.update(config.pruning_fn_default_kwargs)
            kwargs.update(config.prune_layer_types[type(module).__name__])
            if sparsity_map:
                kwargs.update(sparsity_map[0])
            parsed_sparsity_map[name] = kwargs
    return parsed_sparsity_map


def parse_model_for_uniform_pruning(model, config):
    """Parse model for uniform pruning"""
    parsed_explicit_prune = parse_explicit_prune(model, config)
    parsed_not_to_prune = parse_not_to_prune(model, config)
    parsed_sparsity_map = parse_sparsity_map(model, config)
    # remove not to prune layers
    for name in parsed_not_to_prune:
        parsed_sparsity_map.pop(name, None)
    # add the explicit prune layers
    parsed_sparsity_map.update(parsed_explicit_prune)
    return parsed_sparsity_map


def calc_er_k_target_sparsities(model, config):
    """
    Calculate target sparsity ratios per layer with Erdos-Renyi random graph topology
    Based on https://github.com/google-research/rigl/blob/476f56a1999febbdfcf3afc4bf5aae0f4855dd32/rigl/sparse_utils.py#L90
    """
    ignored_layers = set(parse_not_to_prune(model, config))
    dense_layers = set()
    kernel = config.init_strategy == 'erk'
    done = False
    while not done:
        div = .0
        rhs = .0
        raw_p_one = {}
        for name, module in model.named_modules():
            if type(module).__name__ in config.prune_layer_types and name not in ignored_layers:
                tensor = getattr(
                    module, config.prune_layer_types[type(module).__name__]['name'])
                numel = tensor.numel()
                n_zero = numel * \
                    config.pruning_fn_default_kwargs['target_sparsity']
                if name in dense_layers:
                    rhs -= n_zero
                else:
                    rhs += numel - n_zero
                    shape = tensor.shape
                    end = len(shape) if kernel else 2
                    raw_p_one[name] = np.sum(
                        shape[:end] / np.prod(shape[:end]))
                    div += raw_p_one[name] * numel
        eps = rhs / div
        if max(raw_p_one.values()) * eps > 1:
            dense_layers.add(
                sorted(raw_p_one.items(), key=lambda x: x[1], reverse=True)[0][0])
        else:
            if eps < 0:
                raise RuntimeError("Couldn't find ER pruning rations.")
            done = True
    for k, v in raw_p_one.items():
        raw_p_one[k] = 1 - v * eps
    return raw_p_one


def parse_model_for_er_k_pruning(model, config):
    """Parse model per layer target sparsity ratios with Erdos-Renyi random graph topology"""
    if config.weight_sparsity_map:
        logger.warning(
            "In ER/K pruning initialization weight_sparsity_map is ignore: {}".format(config.weight_sparsity_map))
    sparsities = calc_er_k_target_sparsities(model, config)
    parsed_model = {}
    for name, module in model.named_modules():
        if name in sparsities:
            kwargs = {}
            kwargs.update(config.pruning_fn_default_kwargs)
            kwargs.update(config.prune_layer_types[type(module).__name__])
            kwargs.update(target_sparsity=sparsities[name])
            parsed_model[name] = kwargs
    parsed_model.update(parse_explicit_prune(model, config))
    return parsed_model


INIT_STRAT = {
    "uniform": parse_model_for_uniform_pruning,
    "er": parse_model_for_er_k_pruning,
    "erk": parse_model_for_er_k_pruning,
}


def parse_model_for_pruning(model, config):
    """Parse a model and return a dictionary with initialization parameters per module"""
    if config.init_strategy not in INIT_STRAT:
        raise NotImplementedError(
            f"Initialization strategy {config.init_strategy} was not implemented, please use one of the following: {INIT_STRAT.keys()}")
    return INIT_STRAT[config.init_strategy](model, config)
