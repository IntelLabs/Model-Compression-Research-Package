# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test Quantiation Aware Training module
"""
import unittest

import torch
from torch import nn
from torch import quantization as qt

from model_compression_research.quantization.qat import (
    QuantizedLinear,
    FakeQuantize,
    _get_a_n_scale_decomposition,
    _quantize,
    _dequantize,
    _requantize,
)


def get_scale(x, sym=False):
    max_thresh = x.max()
    min_thresh = x.min()
    if sym:
        scale = torch.max(-min_thresh, max_thresh) / 127.5
    else:
        scale = (max_thresh - min_thresh) / 255
    return scale.item()


def get_qparams(x, sym=False):
    scale = get_scale(x, sym)
    if sym:
        zp = 0
    else:
        zp = (-x.min() / scale).round().int().item()
    return scale, zp


REPEATS = 1000


class TestFakeQuantize(unittest.TestCase):
    def setUp(self):
        self.asym_fake_quant = FakeQuantize(
            observer=qt.MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        self.sym_fake_quant = FakeQuantize(
            observer=qt.MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
        self.x = torch.randn(10, 10) + 1

    def test_asym_training_forward(self):
        qx = self.asym_fake_quant(self.x)
        scale = self.asym_fake_quant.scale
        zero_point = self.asym_fake_quant.zero_point
        qx_hat = _dequantize(
            _quantize(self.x, scale, zero_point), scale, zero_point)
        self.assertTrue((qx == qx_hat).all())

    def test_sym_training_forward(self):
        qx = self.sym_fake_quant(self.x)
        scale = self.sym_fake_quant.scale
        zero_point = self.sym_fake_quant.zero_point
        quant_min = self.sym_fake_quant.quant_min
        quant_max = self.sym_fake_quant.quant_max
        qx_hat = _dequantize(
            _quantize(self.x, scale, zero_point, quant_min, quant_max), scale, zero_point)
        self.assertTrue((qx == qx_hat).all())

    def test_asym_inference_forward(self):
        self.asym_fake_quant(self.x)
        scale = self.asym_fake_quant.scale[0]
        zero_point = self.asym_fake_quant.zero_point[0]
        qx = torch.quantize_per_tensor(self.x, scale, zero_point, torch.quint8)
        self.asym_fake_quant.eval()
        qx_hat = self.asym_fake_quant(self.x).to(torch.uint8)
        self.assertTrue((qx.int_repr() == qx_hat).all())

    def test_sym_inference_forward(self):
        self.sym_fake_quant(self.x)
        scale = self.sym_fake_quant.scale[0]
        zero_point = self.sym_fake_quant.zero_point[0]
        qx = torch.quantize_per_tensor(self.x, scale, zero_point, torch.qint8)
        self.sym_fake_quant.eval()
        qx_hat = self.sym_fake_quant(self.x).to(torch.int8)
        self.assertTrue((qx.int_repr() == qx_hat).all())

    def test_quantization_disable(self):
        self.sym_fake_quant.disable_fake_quant()
        x_hat = self.sym_fake_quant(self.x)
        self.assertTrue((self.x == x_hat).all())
        self.sym_fake_quant.enable_fake_quant()
        x_hat = self.sym_fake_quant(self.x)
        self.assertTrue((self.x != x_hat).any())


class TestQuantizedLinear(unittest.TestCase):
    def setUp(self):
        self.weight_asym_fake_quant = FakeQuantize.with_args(
            observer=torch.quantization.MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        )
        self.x = torch.randn(4, 10)
        self.x_scale, self.x_zp = get_qparams(self.x, sym=False)
        self.fqx = torch.fake_quantize_per_tensor_affine(
            self.x, self.x_scale, self.x_zp, 0, 255)
        self.qx = torch.quantize_per_tensor(
            self.x, self.x_scale, self.x_zp, torch.quint8).int_repr().float()

    def test_training_symmetric_weights_no_requant(self):
        for _ in range(REPEATS):
            ql_sym_weight_no_requant = QuantizedLinear(
                10, 6, bias=False, output_fake_quant=None)
            w_scale, _ = get_qparams(ql_sym_weight_no_requant.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weight_no_requant.weight, w_scale, 0, -128, 127)
            self.assertTrue(
                (qw == ql_sym_weight_no_requant.quantized_weight).all())
            y = ql_sym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_symmetric_weights_requant(self):
        for _ in range(REPEATS):
            ql_sym_weights = QuantizedLinear(10, 6, bias=False)
            w_scale, _ = get_qparams(ql_sym_weights.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weights.weight, w_scale, 0, -128, 127)
            self.assertTrue((qw == ql_sym_weights.quantized_weight).all())
            y = ql_sym_weights(self.x)
            y_hat = self.fqx @ qw.t()
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights_no_requant(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QuantizedLinear(10, 6, bias=False,
                                                        output_fake_quant=None, weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QuantizedLinear(10, 6, bias=False,
                                                        weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t()
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_symmetric_weights_with_bias(self):
        for _ in range(REPEATS):
            ql_sym_weights = QuantizedLinear(10, 6)
            w_scale, _ = get_qparams(ql_sym_weights.weight, sym=True)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_sym_weights.weight, w_scale, 0, -128, 127)
            self.assertTrue((qw == ql_sym_weights.quantized_weight).all())
            y = ql_sym_weights(self.x)
            y_hat = self.fqx @ qw.t() + ql_sym_weights.bias
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_training_asymmetric_weights_with_bias(self):
        for _ in range(REPEATS):
            ql_asym_weight_no_requant = QuantizedLinear(
                10, 6, weight_fake_quant=self.weight_asym_fake_quant)
            w_scale, w_zp = get_qparams(ql_asym_weight_no_requant.weight)
            qw = torch.fake_quantize_per_tensor_affine(
                ql_asym_weight_no_requant.weight, w_scale, w_zp, 0, 255)
            self.assertTrue(
                (qw == ql_asym_weight_no_requant.quantized_weight).all())
            y = ql_asym_weight_no_requant(self.x)
            y_hat = self.fqx @ qw.t() + ql_asym_weight_no_requant.bias
            y_scale, y_zp = get_qparams(y_hat)
            y_hat = torch.fake_quantize_per_tensor_affine(
                y_hat, y_scale, y_zp, 0, 255)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_inference_sym_weights_no_requant(self):
        for _ in range(REPEATS):
            ql = QuantizedLinear(10, 6, output_fake_quant=None)
            # run x through the layer once to get statistics
            y_train = ql(self.x)
            # move to eval mode
            ql.eval()
            y = ql(self.x)
            w_scale, w_zp = get_qparams(ql.weight, sym=True)
            qw = torch.quantize_per_tensor(
                ql.weight, w_scale, w_zp, torch.qint8).int_repr().float()
            self.assertTrue(
                (qw == ql.quantized_weight).all())

            input_times_weight_scale = w_scale * self.x_scale
            q_bias = torch.quantize_per_tensor(
                ql.bias, input_times_weight_scale, 0, torch.qint32).int_repr().float()
            y_hat = _dequantize(self.qx @ qw.t() -
                                (0. if w_zp == 0. else w_zp * qx.sum(-1, keepdims=True)) -
                                (0. if self.x_zp == 0. else self.x_zp * qw.t().sum(-2, keepdims=True)) -
                                - self.x_zp * w_zp * ql.in_features + q_bias, input_times_weight_scale)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)
            self.assertTrue((y_train - y).pow(2).mean() < 1e-4)

    def test_inference_sym_weights(self):
        for _ in range(REPEATS):
            ql = QuantizedLinear(10, 6)
            # run x through the layer once to get statistics
            y_train = ql(self.x)
            y_scale, y_zp = get_qparams(y_train)
            self.assertAlmostEqual(y_scale, ql.output_fake_quant.scale.item())
            self.assertTrue(y_zp == ql.output_fake_quant.zero_point)
            # move to eval mode
            ql.eval()
            y = ql(self.x)
            w_scale, w_zp = get_qparams(ql.weight, sym=True)
            qw = torch.quantize_per_tensor(
                ql.weight, w_scale, w_zp, torch.qint8).int_repr().float()
            self.assertTrue(
                (qw == ql.quantized_weight).all())
            input_times_weight_scale = torch.tensor(
                w_scale) * torch.tensor(self.x_scale)
            requant_scale = input_times_weight_scale / torch.tensor(y_scale)
            n, a = _get_a_n_scale_decomposition(requant_scale)
            q_bias = torch.quantize_per_tensor(
                ql.bias, input_times_weight_scale, 0, torch.qint32).int_repr().float()
            y_hat = self.qx @ qw.t() \
                - (0. if w_zp == 0. else w_zp * self.qx.sum(-1, keepdims=True)) \
                - (0. if self.x_zp == 0. else self.x_zp * qw.t().sum(-2, keepdims=True)) \
                + self.x_zp * w_zp * ql.in_features + q_bias
            y_hat = ((y_hat * a) >> n).round() + y_zp
            y_hat = y_hat.clamp(0, 255)
            y_hat = _dequantize(y_hat, y_scale, y_zp)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)
            self.assertTrue((y_train - y).pow(2).mean() < 1e-4)

    def test_inference_sym_weights_no_bias(self):
        for _ in range(REPEATS):
            ql = QuantizedLinear(10, 6, bias=False)
            # run x through the layer once to get statistics
            y_train = ql(self.x)
            y_scale, y_zp = get_qparams(y_train)
            self.assertAlmostEqual(y_scale, ql.output_fake_quant.scale.item())
            self.assertTrue(y_zp == ql.output_fake_quant.zero_point)
            # move to eval mode
            ql.eval()
            w_scale, w_zp = get_qparams(ql.weight, sym=True)
            qw = torch.quantize_per_tensor(
                ql.weight, w_scale, w_zp, torch.qint8).int_repr().float()
            self.assertTrue(
                (qw == ql.quantized_weight).all())
            y = ql(self.x)
            input_times_weight_scale = torch.tensor(
                w_scale) * torch.tensor(self.x_scale)
            requant_scale = input_times_weight_scale / torch.tensor(y_scale)
            n, a = _get_a_n_scale_decomposition(requant_scale)
            y_hat = self.qx @ qw.t() \
                - (0. if w_zp == 0. else w_zp * self.qx.sum(-1, keepdims=True)) \
                - (0. if self.x_zp == 0. else self.x_zp * qw.t().sum(-2, keepdims=True)) \
                + self.x_zp * w_zp * ql.in_features
            y_hat = ((y_hat * a) >> n).round() + y_zp
            y_hat = y_hat.clamp(0, 255)
            y_hat = _dequantize(y_hat, y_scale, y_zp)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)
            self.assertTrue((y_train - y).pow(2).mean() < 1e-4)

    def test_inference_asym_weights(self):
        for _ in range(REPEATS):
            ql = QuantizedLinear(
                10, 6, weight_fake_quant=self.weight_asym_fake_quant)
            # run x through the layer once to get statistics
            y_train = ql(self.x)
            y_scale, y_zp = get_qparams(y_train)
            if torch.abs(y_scale - ql.output_fake_quant.scale) > 1e-7:
                pass
            self.assertAlmostEqual(y_scale, ql.output_fake_quant.scale.item())
            self.assertTrue(y_zp == ql.output_fake_quant.zero_point)
            # move to eval mode
            ql.eval()
            y = ql(self.x)
            w_scale, w_zp = get_qparams(ql.weight)
            qw = torch.quantize_per_tensor(
                ql.weight, w_scale, w_zp, torch.quint8).int_repr().float()
            self.assertTrue(
                (qw == ql.quantized_weight).all())
            input_times_weight_scale = torch.tensor(
                w_scale) * torch.tensor(self.x_scale)
            requant_scale = input_times_weight_scale / torch.tensor(y_scale)
            n, a = _get_a_n_scale_decomposition(requant_scale)
            q_bias = torch.quantize_per_tensor(
                ql.bias, input_times_weight_scale, 0, torch.qint32).int_repr().float()
            y_hat = self.qx @ qw.t() \
                - (0. if w_zp == 0. else w_zp * self.qx.sum(-1, keepdims=True)) \
                - (0. if self.x_zp == 0. else self.x_zp * qw.t().sum(-2, keepdims=True)) \
                + self.x_zp * w_zp * ql.in_features + q_bias
            y_hat = ((y_hat * a) >> n).round() + y_zp
            y_hat = y_hat.clamp(0, 255)
            y_hat = _dequantize(y_hat, y_scale, y_zp)
            self.assertTrue((y_train - y).pow(2).mean() < 1e-4)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)

    def test_inference_asym_weights_no_requant(self):
        for _ in range(REPEATS):
            ql = QuantizedLinear(10, 6, output_fake_quant=None,
                                 weight_fake_quant=self.weight_asym_fake_quant)
            # run x through the layer once to get statistics
            y_train = ql(self.x)
            # move to eval mode
            ql.eval()
            y = ql(self.x)
            w_scale, w_zp = get_qparams(ql.weight)
            qw = torch.quantize_per_tensor(
                ql.weight, w_scale, w_zp, torch.quint8).int_repr().float()
            self.assertTrue(
                (qw == ql.quantized_weight).all())
            input_times_weight_scale = w_scale * self.x_scale
            q_bias = torch.quantize_per_tensor(
                ql.bias, input_times_weight_scale, 0, torch.qint32).int_repr().float()
            y_hat = _dequantize(self.qx @ qw.t() -
                                (0. if w_zp == 0. else w_zp * self.qx.sum(-1, keepdims=True)) -
                                (0. if self.x_zp == 0. else self.x_zp * qw.t().sum(-2, keepdims=True)) +
                                self.x_zp * w_zp * ql.in_features + q_bias, input_times_weight_scale)
            self.assertTrue((y_hat - y).pow(2).mean() < 1e-6)
            self.assertTrue((y_train - y).pow(2).mean() < 1e-4)

    def test_observer_not_collecting_data_when_evaluating(self):
        ql = QuantizedLinear(10, 6)
        for _ in range(3):
            ql(torch.randn(3, 10))
        input_min_val = ql.input_fake_quant.activation_post_process.min_val.item()
        input_max_val = ql.input_fake_quant.activation_post_process.max_val.item()
        output_min_val = ql.output_fake_quant.activation_post_process.min_val.item()
        output_max_val = ql.output_fake_quant.activation_post_process.max_val.item()
        ql.eval()
        for _ in range(3):
            ql(torch.randn(3, 10))
        self.assertTrue(
            input_min_val == ql.input_fake_quant.activation_post_process.min_val.item())
        self.assertTrue(
            input_max_val == ql.input_fake_quant.activation_post_process.max_val.item())
        self.assertTrue(
            output_min_val == ql.output_fake_quant.activation_post_process.min_val.item())
        self.assertTrue(
            output_max_val == ql.output_fake_quant.activation_post_process.max_val.item())
        ql.train()
        for _ in range(3):
            ql(torch.randn(3, 10))
        self.assertFalse(
            input_min_val == ql.input_fake_quant.activation_post_process.min_val.item())
        self.assertFalse(
            input_max_val == ql.input_fake_quant.activation_post_process.max_val.item())
        self.assertFalse(
            output_min_val == ql.output_fake_quant.activation_post_process.min_val.item())
        self.assertFalse(
            output_max_val == ql.output_fake_quant.activation_post_process.max_val.item())

    def test_disable_quantization(self):
        # Training
        ql = QuantizedLinear(10, 6)
        l = nn.Linear(10, 6)
        l.weight.data = ql.weight
        l.bias.data = ql.bias
        y_hat = ql(self.x)
        y = l(self.x)
        self.assertTrue((y != y_hat).any())
        ql.disable_fake_quant()
        ql.disable_observer()
        y_tilde = ql(self.x)
        self.assertTrue((y == y_tilde).all())
        ql.enable_fake_quant()
        y_double_hat = ql(self.x)
        self.assertTrue((y_double_hat == y_hat).all())
        # Not training
        ql.eval()
        y_hat = ql(self.x)
        self.assertTrue(((y_double_hat - y_hat).abs() < 1e-4).all())
        self.assertTrue((y != y_hat).any())
        ql.disable_fake_quant()
        y_tilde = ql(self.x)
        self.assertTrue((y == y_tilde).all())
        ql.enable_fake_quant()
        y_double_hat = ql(self.x)
        self.assertTrue((y_double_hat == y_hat).all())

    def test_delayed_start(self):
        ql = QuantizedLinear(10, 6, start_step=2)
        l = nn.Linear(10, 6)
        l.weight.data = ql.weight
        l.bias.data = ql.bias
        y_hat = ql(self.x)
        y = l(self.x)
        self.assertTrue((y == y_hat).all())
        ql.eval()
        for _ in range(3):
            y_hat = ql(self.x)
        self.assertTrue((y == y_hat).all())
        ql.train()
        y_hat = ql(self.x)
        y_hat = ql(self.x)
        self.assertTrue((y != y_hat).any())
        ql.eval()
        y_hat = ql(self.x)
        self.assertTrue((y != y_hat).any())

    def test_from_float(self):
        l = nn.Linear(10, 6)
        ql = QuantizedLinear.from_float(l)
        self.assertTrue((l.weight == ql.weight).all())
        self.assertTrue((l.bias == ql.bias).all())
        l.bias = None
        ql = QuantizedLinear.from_float(l)
        self.assertTrue((l.weight == ql.weight).all())
        self.assertTrue(ql.bias is None)

    def test_saving_and_loading(self):
        ql = QuantizedLinear(10, 6)
        state_dict = ql.state_dict()
        ql2 = QuantizedLinear(10, 6)
        self.assertTrue((ql.weight != ql2.weight).any())
        ql2.load_state_dict(state_dict)
        self.assertTrue((ql.weight == ql2.weight).all())
        self.assertTrue((ql.bias == ql2.bias).all())


if __name__ == "__main__":
    unittest.main()
