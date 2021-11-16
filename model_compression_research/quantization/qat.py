# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Quantization ops
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import quantization as tq


def _quantize(input, scale=1., zero_point=0, quant_min=0, quant_max=255):
    """Do linear quantization to input according to a scale and number of bits"""
    return input.mul(1 / scale).round().add(zero_point).clamp(quant_min, quant_max)


def _dequantize(input, scale=1., zero_point=0):
    """linear dequantization according to some scale"""
    return input.sub(zero_point).mul(scale)


def _get_a_n_scale_decomposition(scale, scale_bits=16):
    n = (scale_bits - 1 - torch.log2(scale.abs())).floor()
    a = (scale.abs() * 2 ** n).round().clamp(0,
                                             calc_max_quant_value(scale_bits))
    return n, a


def _requantize(input, input_scale=1., input_zero_point=0, output_scale=1., output_zero_point=0, quant_min=0, quant_max=255, scale_bits=16):
    if input_zero_point != 0:
        raise NotImplementedError(
            "Requantization is not implemented yet for assymetric input")
    scale = input_scale / output_scale
    n, a = _get_a_n_scale_decomposition(scale, scale_bits)
    out = ((input * a) >> n).round() + output_zero_point
    out = out.clamp(quant_min,
                    quant_max)
    return out


class FakeQuantize(tq.FakeQuantize):
    def forward(self, X):
        if (not self.training) and self.fake_quant_enabled[0] == 1:
            X = self.quantize(X)
        else:
            X = super().forward(X)
        return X

    def quantize(self, X):
        X = _quantize(X, self.scale, self.zero_point,
                      self.quant_min, self.quant_max)
        return X


default_activation_fake_quant = FakeQuantize.with_args(
    observer=tq.MovingAverageMinMaxObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine
)


default_weight_fake_quant = FakeQuantize.with_args(
    observer=tq.MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_tensor_symmetric
)


def calc_max_quant_value(bits):
    """Calculate the maximum symmetric quantized value according to number of bits"""
    return 2**(bits) - 1


class QuantizedLinear(nn.Linear):
    """Linear layer with quantization aware training capability"""

    class QuantizedLinearOutput(FakeQuantize):
        def __init__(self, output_fake_quant=None):
            requantize = output_fake_quant is not None
            if requantize:
                super().__init__(*output_fake_quant.p.args, **output_fake_quant.p.keywords)
            else:
                super().__init__()
                self.fake_quant_enabled[0] = 0
                self.observer_enabled[0] = 0
            self.register_buffer('requantize', torch.tensor(
                [requantize], dtype=torch.uint8))
            self.register_buffer(
                'quant_enabled', torch.ones(1, dtype=torch.uint8))

        def forward(self, X, weight_times_input_scale=None):
            if (not self.training) and self.quant_enabled[0] == 1:
                out = self.quant_infer(X, weight_times_input_scale)
            else:
                out = super().forward(X)
            return out

        def quant_infer(self, X, weight_times_input_scale):
            assert not self.training, "This method should only be called when doing evaluation"
            assert weight_times_input_scale is not None, "This is a required parameter for this function"
            out_scale = weight_times_input_scale
            out_zero_point = 0
            out = X
            if self.requantize[0]:
                out_zero_point = self.zero_point
                out_scale = self.scale
                out = _requantize(out, weight_times_input_scale, 0, out_scale,
                                  out_zero_point, self.quant_min, self.quant_max)
            out = _dequantize(out, out_scale, out_zero_point)
            return out

        def enable_fake_quant(self, enabled=True):
            self.quant_enabled[0] = 1 if enabled else 0
            if self.requantize[0]:
                super().enable_fake_quant(enabled)

        def enable_observer(self, enabled=True):
            if self.requantize[0]:
                super().enable_observer(enabled)

        def extra_repr(self):
            return 'requantize={}, '.format(self.requantize) + super().extra_repr()

    def __init__(self, in_features, out_features, bias=True, start_step=0,
                 weight_fake_quant=default_weight_fake_quant,
                 input_fake_quant=default_activation_fake_quant,
                 output_fake_quant=default_activation_fake_quant
                 ):
        super().__init__(in_features, out_features, bias)
        self.accumulation_bits = 32
        self.start_step = int(start_step)
        self.weight_fake_quant = weight_fake_quant()
        self.input_fake_quant = input_fake_quant()
        self.output_fake_quant = self.QuantizedLinearOutput(output_fake_quant)
        self.register_buffer('_step', torch.zeros(1))
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8))
        if self.start_step > 0:
            self.disable_observer()
            self.disable_fake_quant()

    @classmethod
    def from_float(cls, module, start_step=0,
                   weight_fake_quant=default_weight_fake_quant,
                   input_fake_quant=default_activation_fake_quant,
                   output_fake_quant=default_activation_fake_quant
                   ):
        new = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            start_step=start_step,
            weight_fake_quant=weight_fake_quant,
            input_fake_quant=input_fake_quant,
            output_fake_quant=output_fake_quant
        )
        new.weight.data = module.weight
        if module.bias is not None:
            new.bias.data = module.bias
        return new

    def training_quantized_forward(self, input):
        """fake quantized forward, fake quantizes weights and activations,
        learn quantization ranges if quantization mode is EMA.
        This function should only be used while training"""
        return self.output_fake_quant(
            F.linear(self.input_fake_quant(input), self.quantized_weight, self.bias))

    def inference_quantized_forward(self, input):
        """Simulate quantized inference. quantize input and perform calculation with only integer numbers.
        This function should only be used while doing inference"""
        assert not self.training, "should only be called when not training"
        q_input = self.input_fake_quant(input)
        return self.output_fake_quant(
            F.linear(q_input, self.quantized_weight, self.quantized_bias) -
            self.weight_fake_quant.zero_point * q_input.sum(-1, keepdims=True),
            self.input_fake_quant.scale * self.weight_fake_quant.scale
        )

    def forward(self, input):
        if self.training:
            if self._step == self.start_step:
                self.enable_fake_quant()
                self.enable_observer()
            self._step += 1
        if (not self.training) and self.fake_quant_enabled[0] == 1:
            out = self.inference_quantized_forward(input)
        else:
            out = self.training_quantized_forward(input)
        return out

    @property
    def quantized_bias(self):
        bias = -self.input_fake_quant.zero_point * \
            self.quantized_weight.t().sum(-2, keepdims=True)
        bias += self.input_fake_quant.zero_point * \
            self.weight_fake_quant.zero_point * self.in_features
        try:
            n = calc_max_quant_value(self.accumulation_bits - 1)
            bias += _quantize(self.bias, self.input_fake_quant.scale *
                              self.weight_fake_quant.scale, 0, -n, n - 1)
        except AttributeError:
            # bias is None, we dont need to do anything further
            pass
        return bias if (bias != torch.zeros_like(bias)).any() else None

    @property
    def quantized_weight(self):
        return self.weight_fake_quant(self.weight)

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self.input_fake_quant.enable_fake_quant(enabled)
        self.weight_fake_quant.enable_fake_quant(enabled)
        self.output_fake_quant.enable_fake_quant(enabled)
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.input_fake_quant.enable_observer(enabled)
        self.weight_fake_quant.enable_observer(enabled)
        self.output_fake_quant.enable_observer(enabled)
        return self

    def disable_observer(self):
        return self.enable_observer(False)


QUANT_MAPPING = {
    nn.Linear: QuantizedLinear
}

UNQUANT_MAPPING = {
    QuantizedLinear: nn.Linear
}
