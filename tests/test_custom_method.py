# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Testing Custom based method
"""
import unittest

import torch
from torch import nn

from model_compression_research import (custom_mask_pruning,
                                        PruningMethod,
                                        remove_pruning,
                                        )


class TestCustomMaskPruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(3, 4)
        weight = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.],
                               [1., 2., 3.]])
        linear.weight.data = weight
        true_mask = torch.tensor([[1., 0., 1.],
                                  [0., 0., 1.],
                                  [1., 0., 0.],
                                  [0., 1., 0.]])
        masked_weight = torch.tensor([[1., 0., 3.],
                                      [0., 0., 6.],
                                      [7., 0., 0.],
                                      [0., 2., 0.]])
        linear = custom_mask_pruning(linear, mask=true_mask)[0]
        original, mask, method = linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(type(linear.weight) is torch.Tensor)
        self.assertTrue(
            (original == weight).all())
        self.assertTrue((mask == true_mask).all())
        self.assertTrue((linear.weight == masked_weight).all())

        def check_pruning_method(module):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    return True
            return False
        self.assertTrue(check_pruning_method(linear))

    def test_pruning_method_remove(self):
        linear = nn.Linear(10, 10)
        mask = (torch.rand_like(linear.weight) > 0.5).float()
        linear, method = custom_mask_pruning(linear, mask=mask)
        remove_pruning(linear)
        self.assertFalse(hasattr(linear, method.get_name('original')))
        self.assertTrue(type(linear.weight) is nn.Parameter)
        self.assertFalse(hasattr(linear, method.get_name('mask')))
        self.assertFalse(hasattr(linear, method.get_name('method')))

        def check_pruning_method(module):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    return True
            return False
        self.assertFalse(check_pruning_method(linear))

    def test_pruning_gradients(self):
        linear = custom_mask_pruning(nn.Linear(10, 5), mask=(
            torch.rand(10, 5).t() > 0.5).float())[0]
        model = linear
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = nn.DataParallel(model)
            model.to(device)
        x = torch.rand((4, 10)).to(device)
        y = model(x).sum()
        y.backward()
        original, mask = linear.get_pruning_parameters('original', 'mask')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad).masked_fill(~mask.bool(), 0.)).pow(2).mean().cpu().item(), 0.)


if __name__ == "__main__":
    unittest.main()
