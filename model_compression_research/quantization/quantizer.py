# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Quantizer, apply quantization modules to supported pytorch models
"""
from ..utils import Config
from . import qat


def get_unique_devices(module):
    return {p.device for p in module.parameters()} | {p.device for p in module.buffers()}


class QuantizerConfig(Config):
    ATTRIBUTES = {
        "quantization_begin": 0,
        "not_to_quantize": [],
        "not_to_requantize_output": []
    }


class Quantizer:
    def __init__(self, model, config):
        if type(config) != QuantizerConfig:
            raise TypeError("Expected quantizer config from type {}, got type{}".format(
                Quantizer.__name__, type(config).__name__))
        self.model = model
        self.config = config

    def _quantize(self, module, path=''):
        swap_dict = {}
        for name, mod in module.named_children():
            full_name = '.'.join((path, name)).strip('.')
            if type(mod) not in qat.QUANT_MAPPING:
                self._quantize(mod, full_name)
            else:
                if not any([s in full_name for s in self.config.not_to_quantize]):
                    if any([s in full_name for s in self.config.not_to_requantize_output]):
                        new = qat.QUANT_MAPPING[type(mod)].from_float(
                            mod, start_step=self.config.quantization_begin, output_fake_quant=None)
                    else:
                        new = qat.QUANT_MAPPING[type(mod)].from_float(
                            mod, start_step=self.config.quantization_begin)
                    devices = get_unique_devices(mod)
                    assert len(
                        devices) <= 1, ("swap_module only works with cpu or single-device CUDA modules, but got devices {}".format(devices))
                    swap_dict[name] = new.to(
                        next(iter(devices)) if len(devices) > 0 else None)
        for key, value in swap_dict.items():
            module._modules[key] = value

    def quantize(self):
        self._quantize(self.model)
        return self.model
