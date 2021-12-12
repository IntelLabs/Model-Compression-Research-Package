# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


def sample_length_configuration(
    max_seq_length,
    num_hidden_layers,
    layer_config=None,
    length_drop_prob=None,
    length_drop_ratio=None,
    length_drop_ratio_bound=None,
    min_length=2,
):
    length = max_seq_length
    length_configuration = ()
    for i in range(num_hidden_layers):
        if layer_config is None or i in layer_config:
            if length_drop_prob is not None:
                length = length - np.random.binomial(length, length_drop_prob)
            elif length_drop_ratio is not None:
                length = int(np.ceil(length * (1 - length_drop_ratio)))
            elif length_drop_ratio_bound is not None:
                length = np.random.randint(int(np.ceil(length * (1 - length_drop_ratio_bound))), length + 1)
        length = max(length, min_length)
        length_configuration += (length,)
    return length_configuration


def sample_layer_configuration(
    num_hidden_layers,
    layer_dropout_prob=None,
    layer_dropout=None,
    layer_dropout_bound=None,
):
    if layer_dropout_prob is not None:
        return tuple(i for i in range(num_hidden_layers) if np.random.random() >= layer_dropout_prob)
    elif layer_dropout is not None:
        layer_dropout = min(layer_dropout, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - layer_dropout))
    elif layer_dropout_bound is not None:
        layer_dropout_bound = min(layer_dropout_bound, num_hidden_layers - 1)
        return tuple(range(num_hidden_layers - np.random.randint(0, layer_dropout_bound + 1)))
    return None


@dataclass
class LengthDropArguments:
    length_config: Optional[List[int]] = None
    length_adaptive: Optional[bool] = False
    num_sandwich: Optional[int] = 2
    length_drop_ratio_bound: Optional[float] = 0.2
    layer_dropout_prob: Optional[float] = 0.2
    layer_dropout_bound: Optional[int] = 0


def add_drop_and_restore_args(parser):
    parser.add_argument(
        "--length_config",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--length_adaptive",
        action="store_true",
    )
    parser.add_argument(
        "--num_sandwich",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--length_drop_ratio_bound",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--layer_dropout_prob",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--layer_dropout_bound",
        default=0,
        type=int,
    )


@dataclass
class SearchArguments:
    do_search: Optional[bool] = field(default=False)
    load_store_file: Optional[str] = field(default=None)
    evo_iter: Optional[int] = field(default=100)
    population_size: Optional[int] = field(default=20)
    mutation_size: Optional[int] = field(default=30)
    mutation_prob: Optional[float] = field(default=0.5)
    crossover_size: Optional[int] = field(default=30)


def add_search_args(parser):
    parser.add_argument("--do_search", action="store_true")
    parser.add_argument("--load_store_file", default=None, type=str)
    parser.add_argument("--evo_iter", default=100, type=int)
    parser.add_argument("--population_size", default=20, type=int)
    parser.add_argument("--mutation_size", default=30, type=int)
    parser.add_argument("--mutation_prob", default=0.5, type=float)
    parser.add_argument("--crossover_size", default=30, type=int)
