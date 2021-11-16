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
    unstructured_magnitude_pruning,
    block_structured_magnitude_pruning,
    UnstructuredSparsityGroup,
    grouped_unstructured_magnitude_pruning,
    PruningMethod,
    remove_pruning,
    get_tensor_sparsity_ratio,
    uniform_magnitude_pruning,
)


class TestMagnitudePruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(10, 10)
        weight = linear.weight.data
        linear = unstructured_magnitude_pruning(linear, threshold_decay=0.8)[0]
        original, mask, method = linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(type(linear.weight) is torch.Tensor)
        self.assertTrue((original == weight).all())
        self.assertTrue((mask == torch.ones_like(
            weight)).all())
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
        linear, method = unstructured_magnitude_pruning(
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
        linear = unstructured_magnitude_pruning(
            nn.Linear(100, 100), threshold_decay=0.)[0]
        weight = linear.get_pruning_parameters('original').data
        self.assertEqual(get_tensor_sparsity_ratio(linear.weight), 0.)
        sparsity = 0.5
        unstructured_magnitude_pruning(linear, target_sparsity=sparsity)
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
        # torch.manual_seed(0)
        linear = unstructured_magnitude_pruning(
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
        original, mask = linear.get_pruning_parameters('original', 'mask')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad).masked_fill(~mask.bool(), 0.)).pow(2).mean().cpu().item(), 0.)


class TestUniformMagnitudePruningMethod(unittest.TestCase):
    def setUp(self):
        self.linear = nn.Linear(8, 8)
        self.weight = torch.tensor([
            [8.5365, -5.1200, 8.5507, 3.4815, 4.8626, -2.7110, 8.5555, 6.4250],
            [-.5373, -7.7090, -1.4797, -4.2996, 6.4628, -3.0498, -2.1962, 1.8144],
            [-3.1835, 8.7985, 9.8323, 4.5388, -7.1623, 2.2442, -3.9784, -7.1318],
            [5.8405, 4.5364, 9.7286, -3.1511, 2.8932, 1.0971, 1.9512, -5.7185],
            [4.6720, -9.7932, 4.6963, 8.7403, 7.7966, -7.5372, 3.0610, -7.8093],
            [-2.3322, 8.6801, 0.2020, 2.0883, -1.9596, -9.6593, -7.9417, 9.7893],
            [7.8499, -6.6032, 5.5826, 7.1758, -8.4269, -9.6372, 5.0284, -2.4561],
            [0.2173, -8.6424, -3.4630, 5.2946, -6.1135, -0.7720, -5.1937, 2.3105],
        ])
        self.linear.weight.data = self.weight

    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        uniform_magnitude_pruning(self.linear)
        original, mask, method = self.linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(self.linear, 'weight'))
        self.assertTrue(type(self.linear.weight) is torch.Tensor)
        self.assertTrue((original == self.weight).all())
        self.assertTrue((mask == torch.ones_like(
            self.weight, dtype=torch.bool)).all())
        self.assertTrue((self.linear.weight == self.weight).all())

        def check_pruning_method(module):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    return True
            return False
        self.assertTrue(check_pruning_method(self.linear))

    def test_pruning_method_remove(self):
        _, method = uniform_magnitude_pruning(self.linear)
        remove_pruning(self.linear)
        self.assertFalse(hasattr(self.linear, method.get_name('original')))
        self.assertTrue(type(self.linear.weight) is nn.Parameter)
        self.assertFalse(hasattr(self.linear, method.get_name('mask')))
        self.assertFalse(hasattr(self.linear, method.get_name('method')))

        def check_pruning_method(module):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    return True
            return False
        self.assertFalse(check_pruning_method(self.linear))

    def test_masking(self):
        uniform_magnitude_pruning(self.linear, block_size=4)
        self.assertEqual(get_tensor_sparsity_ratio(self.linear.weight), 0.)
        sparsity = 0.5
        uniform_magnitude_pruning(self.linear, target_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(self.linear.weight), sparsity)
        mask = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 1, 0],
        ])
        self.assertTrue(
            (mask == self.linear.get_pruning_parameters('mask')).all())
        sparsity = 0.75
        uniform_magnitude_pruning(self.linear, target_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(self.linear.weight), sparsity)
        mask = torch.tensor([
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
        ])
        self.assertTrue(
            (mask == self.linear.get_pruning_parameters('mask')).all())
        tensor = self.linear.weight.clone()
        remove_pruning(self.linear)
        self.assertTrue((tensor == self.linear.weight).all())

    def test_pruning_gradients(self):
        sparsity = .5
        # torch.manual_seed(0)
        uniform_magnitude_pruning(self.linear, initial_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(self.linear.weight), sparsity)
        model = self.linear
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = nn.DataParallel(model)
            model.to(device)
        x = torch.rand((4, 8)).to(device)
        y = model(x).sum()
        y.backward()
        original, mask = self.linear.get_pruning_parameters('original', 'mask')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad).masked_fill(~mask.bool(), 0.)).pow(2).mean().cpu().item(), 0.)


class TestUnstructuredSparsityGroup(unittest.TestCase):
    def test_calc_new_threshold(self):
        l1 = nn.Linear(100, 100)
        l2 = nn.Linear(100, 100)
        l3 = nn.Linear(100, 100)
        # since the modules init with the same distributions we want to change the dist a bit
        l1.weight.data = l1.weight * 1.5
        l3.weight.data = l3.weight * 0.5
        _, m1 = unstructured_magnitude_pruning(l1)
        _, m2 = unstructured_magnitude_pruning(l2)
        _, m3 = unstructured_magnitude_pruning(l3)
        group = UnstructuredSparsityGroup()
        group.add(m1)
        group.add(m2)
        group.add(m3)
        self.assertTrue(len(group.method_list) == 3)
        self.assertAlmostEqual(group.compute_new_threshold(), 0.)
        group.target_sparsity = 0.2
        threshold = group.compute_new_threshold()
        sparsity = []
        for m in [m1, m2, m3]:
            new_mask = (m.get_parameters('original').abs()
                        > threshold).byte()
            # print(new_mask)
            m.set_parameter('mask', new_mask)
            setattr(m.module, m.name, m.masked_weight(m.module))
            sparsity.append(get_tensor_sparsity_ratio(
                getattr(m.module, m.name)))
        avg_sparsity = sum(sparsity) / len(sparsity)
        self.assertTrue(avg_sparsity > group.target_sparsity -
                        1 and avg_sparsity < group.target_sparsity + 1)


class TestGroupedUnstructuredMagnitudePruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(10, 10)
        weight = linear.weight.data
        linear = grouped_unstructured_magnitude_pruning(linear)[0]
        original, mask, method = linear.get_pruning_parameters(
            'original', 'mask', 'method')
        self.assertTrue(isinstance(method, PruningMethod))
        self.assertTrue(hasattr(linear, 'weight'))
        self.assertTrue(type(linear.weight) is torch.Tensor)
        self.assertTrue((original == weight).all())
        self.assertTrue((mask == torch.ones_like(
            weight, dtype=torch.bool)).all())
        self.assertTrue((linear.weight == weight).all())

        def check_pruning_method(module):
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, PruningMethod) and hook.name == 'weight':
                    return True
            return False
        self.assertTrue(check_pruning_method(linear))
        remove_pruning(linear)

    def test_pruning_method_remove(self):
        linear = nn.Linear(10, 10)
        linear, method = grouped_unstructured_magnitude_pruning(linear)
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
        # linear = init_grouped_unstructured_magnitude_pruning(nn.Linear(100, 100))[0]
        linear = nn.Linear(3, 2)
        weight = torch.tensor([[1, 2, 3],
                               [4, 5, 6]]).float()
        linear.weight.data = weight
        grouped_unstructured_magnitude_pruning(linear)
        self.assertEqual(get_tensor_sparsity_ratio(linear.weight), 0.)
        sparsity = 0.5
        grouped_unstructured_magnitude_pruning(group_target_sparsity=sparsity)
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

    def test_group_masking(self):
        l1 = nn.Linear(3, 2)
        w1 = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.]])
        l1.weight.data = w1
        l2 = nn.Linear(3, 2)
        w2 = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.5]]) * 0.5
        l2.weight.data = w2
        l3 = nn.Linear(3, 2)
        w3 = torch.tensor([[1., 2., 3.],
                           [4., 5., 6.]]) * 2.
        l3.weight.data = w3
        grouped_unstructured_magnitude_pruning(l1)
        grouped_unstructured_magnitude_pruning(l2)
        grouped_unstructured_magnitude_pruning(l3)
        self.assertEqual(get_tensor_sparsity_ratio(l1.weight), 0.)
        self.assertEqual(get_tensor_sparsity_ratio(l2.weight), 0.)
        self.assertEqual(get_tensor_sparsity_ratio(l3.weight), 0.)
        grouped_unstructured_magnitude_pruning(group_target_sparsity=.5)
        self.assertAlmostEqual(get_tensor_sparsity_ratio(l1.weight), 0.5)
        self.assertAlmostEqual(get_tensor_sparsity_ratio(l2.weight), 5 / 6)
        self.assertAlmostEqual(get_tensor_sparsity_ratio(l3.weight), 1 / 6)
        for l in [l1, l2, l3]:
            remove_pruning(l)

    def test_pruning_gradients(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        sparsity = .75
        linear = grouped_unstructured_magnitude_pruning(nn.Linear(40, 30))[
            0]
        model = linear
        model = nn.DataParallel(model)
        model.to(device)
        grouped_unstructured_magnitude_pruning(group_target_sparsity=sparsity)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(linear.weight), sparsity, 1)
        x = torch.rand((4, 40)).to(device)
        y = model(x).sum()
        y.backward()
        original, mask = linear.get_pruning_parameters('original', 'mask')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad).masked_fill(~mask.bool(), 0.)).pow(2).mean().cpu().item(), 0.)


class TestBlockMagnitudePruningMethod(unittest.TestCase):
    def test_pruning_method_parameters(self):
        """Test that applying the pruning method creates all the parameters it
        need to create in the host layer"""
        linear = nn.Linear(10, 10)
        weight = linear.weight.data
        linear = block_structured_magnitude_pruning(
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
        linear, method = block_structured_magnitude_pruning(
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
        linear = block_structured_magnitude_pruning(
            nn.Linear(100, 100), threshold_decay=0., block_dims=2)[0]
        self.assertEqual(get_tensor_sparsity_ratio(linear.weight), 0.)
        sparsity = 0.5
        block_structured_magnitude_pruning(linear, target_sparsity=sparsity)
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
        block_structured_magnitude_pruning(
            linear, initial_sparsity=0.5, block_dims=(3, 2), pooling_type='max')
        self.assertTrue((mask == linear.get_pruning_parameters('mask')).all())
        tensor = linear.weight.clone()
        remove_pruning(linear)
        self.assertTrue((tensor == linear.weight).all())

    def test_pruning_gradients(self):
        sparsity = .6
        # torch.manual_seed(0)
        linear = block_structured_magnitude_pruning(
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
        original, mask = linear.get_pruning_parameters('original', 'mask')
        self.assertAlmostEqual((original.grad - x.sum(0).expand_as(
            original.grad).masked_fill(~mask.bool(), 0.)).pow(2).mean().cpu().item(), 0.)


if __name__ == '__main__':
    unittest.main()
