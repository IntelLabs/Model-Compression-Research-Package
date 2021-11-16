# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Testing magnitude based method
"""
import unittest

import torch
from torch import nn

from model_compression_research import (
    unstructured_neural_wiring_pruning,
    block_structured_neural_wiring_pruning,
    PruningMethod,
    remove_pruning,
    get_tensor_sparsity_ratio,
)


class TestDNWPruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(10, 10)
        weight = linear.weight.data
        linear = unstructured_neural_wiring_pruning(
            linear, threshold_decay=0.8)[0]
        original, mask, method = linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(type(linear.weight) is torch.Tensor)
        self.assertTrue((original == weight).all())
        self.assertTrue((mask == torch.ones_like(
            weight, dtype=torch.bool)).all())
        self.assertTrue((linear.weight == weight).all())

        def check_pruning_method(module, threshold_decay):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    if hook.threshold_decay == threshold_decay:
                        return True
            return False
        self.assertTrue(check_pruning_method(linear, 0.8))
        method.threshold_decay = 0.7
        self.assertTrue(check_pruning_method(linear, 0.7))

    def test_pruning_method_remove(self):
        linear = nn.Linear(10, 10)
        linear, method = unstructured_neural_wiring_pruning(
            linear, threshold_decay=0.8)
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

    def test_masking(self):
        linear = unstructured_neural_wiring_pruning(
            nn.Linear(100, 100), threshold_decay=0.)[0]
        weight = linear.get_pruning_parameters('original').data
        self.assertEqual(get_tensor_sparsity_ratio(linear.weight), 0.)
        sparsity = 0.5
        unstructured_neural_wiring_pruning(linear, target_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(linear.weight), sparsity)
        # compute mask
        threshold = weight.flatten().abs().sort(
        )[0][int(sparsity * weight.numel()) - 1].item()
        mask = weight.abs() > threshold
        self.assertTrue((mask == linear.get_pruning_parameters('mask')).all())
        tensor = linear.weight.clone()
        remove_pruning(linear)
        self.assertTrue((tensor == linear.weight).all())

    def test_pruning_gradients(self):
        sparsity = .7
        linear = unstructured_neural_wiring_pruning(
            nn.Linear(10, 5), initial_sparsity=sparsity, threshold_decay=0.)[0]
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(linear.weight), sparsity)
        model = linear
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = nn.DataParallel(model)
            model.to(device)
        x = torch.rand((4, 10)).to(device)
        y = model(x).sum()
        y.backward()
        original = linear.get_pruning_parameters('original')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad)).pow(2).mean().cpu().item(), 0.)


class TestBlockDNWPruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(10, 10)
        weight = linear.weight.data
        linear = block_structured_neural_wiring_pruning(
            linear, threshold_decay=0.8)[0]
        original, mask, method = linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(type(linear.weight) is torch.Tensor)
        self.assertTrue((original == weight).all())
        self.assertTrue((mask == torch.ones_like(
            weight, dtype=torch.bool)).all())
        self.assertTrue((linear.weight == weight).all())

        def check_pruning_method(module, threshold_decay):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    if hook.threshold_decay == threshold_decay:
                        return True
            return False
        self.assertTrue(check_pruning_method(linear, 0.8))
        method.threshold_decay = 0.7
        self.assertTrue(check_pruning_method(linear, 0.7))

    def test_pruning_method_remove(self):
        linear = nn.Linear(10, 10)
        linear, method = block_structured_neural_wiring_pruning(
            linear, threshold_decay=0.8)
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

    def test_sparsity_ratio(self):
        linear = block_structured_neural_wiring_pruning(
            nn.Linear(100, 100), threshold_decay=0., block_dims=2)[0]
        self.assertEqual(get_tensor_sparsity_ratio(linear.weight), 0.)
        sparsity = 0.5
        block_structured_neural_wiring_pruning(
            linear, target_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(linear.weight), sparsity)

    def test_masking(self):
        linear = nn.Linear(6, 4)
        weight = torch.tensor([[1, 5, 1, 1, 1, 1],
                               [1, 1, 1, 1, -2, 1],
                               [1, 1, 1, 1, 1, -3],
                               [1, 1, 1, 1, 1, 1],
                               ]).float()
        mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                             [1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1]])
        linear.weight.data = weight
        block_structured_neural_wiring_pruning(
            linear, initial_sparsity=0.5, block_dims=(3, 2), pooling_type='max')
        self.assertTrue((mask == linear.get_pruning_parameters('mask')).all())
        tensor = linear.weight.clone()
        remove_pruning(linear)
        self.assertTrue((tensor == linear.weight).all())

    def test_pruning_gradients(self):
        sparsity = .6
        # torch.manual_seed(0)
        linear = block_structured_neural_wiring_pruning(
            nn.Linear(10, 6), initial_sparsity=sparsity, threshold_decay=0., block_dims=2)[0]
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(linear.weight), sparsity)
        model = linear
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = nn.DataParallel(model)
            model.to(device)
        x = torch.rand((4, 10)).to(device)
        y = model(x).sum()
        y.backward()
        original = linear.get_pruning_parameters('original')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad)).pow(2).mean().cpu().item(), 0.)


if __name__ == "__main__":
    unittest.main()
