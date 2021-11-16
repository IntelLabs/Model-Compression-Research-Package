# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test quantizer
"""
import unittest

import torch
from torch import nn

from model_compression_research.quantization import Quantizer, QuantizerConfig, QuantizedLinear


class TestQuantizer(unittest.TestCase):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128, bias=False)
            self.fc2 = nn.Linear(128, 10, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    def setUp(self):
        self.model = self.Net()
        self.config = QuantizerConfig()

    def test_simple_case(self):
        Quantizer(self.model, self.config).quantize()
        self.assertTrue(type(self.model.fc1) == QuantizedLinear)
        self.assertTrue(type(self.model.fc2) == QuantizedLinear)

    def test_quantization_begin(self):
        self.config.quantization_begin = 2
        Quantizer(self.model, self.config).quantize()
        self.assertTrue(self.model.fc1.start_step == 2)
        self.assertTrue(self.model.fc2.start_step == 2)

    def test_not_to_quantize(self):
        self.config.not_to_quantize = ['fc1']
        Quantizer(self.model, self.config).quantize()
        self.assertTrue(type(self.model.fc1) == nn.Linear)
        self.assertTrue(type(self.model.fc2) == QuantizedLinear)

    def test_not_to_requant(self):
        self.config.not_to_requantize_output = ['fc1']
        Quantizer(self.model, self.config).quantize()
        self.assertTrue(self.model.fc1.output_fake_quant.requantize[0] == 0)
        self.assertTrue(self.model.fc2.output_fake_quant.requantize[0] == 1)


if __name__ == "__main__":
    unittest.main()
