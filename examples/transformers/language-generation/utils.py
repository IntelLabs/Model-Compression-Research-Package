# Apache v2 license
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities file for generation wrapper and model benchmarking
"""
import logging
import time
import json
import os
import copy

import torch


logger = logging.getLogger(__name__)


class CacheConfig:

    string_to_class = {
        'tuple': tuple,
        'list': list,
    }

    default_name = 'cache_config.json'

    def __init__(self, config_list=None):
        self._config_list = copy.deepcopy(config_list)

    @classmethod
    def from_json_file(cls, path):
        with open(path) as f:
            c = cls(json.load(f))
        return c

    @classmethod
    def from_pretrained(cls, path):
        try:
            c = cls.from_json_file(os.path.join(path, cls.default_name))
        except FileNotFoundError:
            logger.error(f"File {cls.default_name} not found in {path=}")
            raise
        return c

    def generate_past_key_values(self,
                                 past_length=0,
                                 batch_size=1,
                                 config_list=None,
                                 dtype=torch.bfloat16,
                                 ):
        if config_list is None:
            config_list = copy.deepcopy(self._config_list)
        else:
            config_list = copy.deepcopy(config_list)
        assert config_list is not None, "Config list not given"
        assert config_list[-1]['type'] == 'Tensor', "Last level of cache has to be a tensor"
        # handle tensor shape
        tensor_shapes = []
        for shape in config_list[-1]['size']:
            tensor_shape = []
            for l in shape:
                if l == 'batch_size':
                    tensor_shape.append(batch_size)
                elif l == 'past_length':
                    tensor_shape.append(past_length)
                else:
                    assert type(
                        l) == int, "Only integer numbers are accepted for tensor shape"
                    tensor_shape.append(l)
            tensor_shapes.append(tuple(tensor_shape))
        config_list[-1]['size'] = tensor_shapes

        def construct_cache(config_list):
            class_name, shape = config_list[0]['type'], config_list[0]['size']
            if class_name == 'Tensor':
                if len(shape) > 1:
                    return tuple(torch.zeros(s) for s in shape)
                else:
                    return torch.zeros(shape[0], dtype=dtype)
            assert class_name in self.string_to_class, f"Unsupported class: {class_name}"
            assert type(
                shape) == int, "shape must be integer for non Tensor classes"
            return self.string_to_class[class_name](construct_cache(config_list[1:]) for _ in range(shape))

        return construct_cache(config_list)


def generate_sample_config(
        batch_size,
        sequence_length,
        past_kv_length=None,
        max_random=10,
        kv_cache_enable=False,
        cache_config=None,
):
    if kv_cache_enable and past_kv_length is None:
        past_kv_length = sequence_length - 1
    if past_kv_length is not None:
        sample = {
            'input_ids': torch.randint(0, max_random, size=(batch_size, sequence_length - past_kv_length), dtype=torch.int64),
            'attention_mask': torch.ones(batch_size, sequence_length, dtype=torch.int64),
            'past_key_values': cache_config.generate_past_key_values(past_kv_length, batch_size)
        }
    else:
        sample = {'input_ids': torch.randint(
            0, max_random, size=(batch_size, sequence_length), dtype=torch.int64)}
    return sample


class Timer:
    UNITS_FACTORS = {
        'seconds': 1,
        'miliseconds': 1000,
    }

    def __init__(self, description='', *, units='miliseconds', verbose=False):
        if units not in self.UNITS_FACTORS:
            raise RuntimeError(
                f'Timer does not support {units}, please select units from list: {list(self.UNITS_FACTORS.keys())}')
        self.description = description
        self.verbose = verbose
        self.units = units

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.time = (time.time() - self.time) * self.UNITS_FACTORS[self.units]
        if self.verbose:
            logger.info(
                f'Task: {self.description} took: {self.time:.3f} {self.units}')
