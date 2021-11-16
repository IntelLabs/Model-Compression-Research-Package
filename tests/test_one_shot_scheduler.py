# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test One Shot Scheduler
"""
import logging
import unittest

import torch
from torch import nn

from model_compression_research import (
    get_tensor_sparsity_ratio,
    OneShotPruningConfig,
    OneShotPruningScheduler,
)


class TestOneShotPruningScheduler(unittest.TestCase):
    def test_pruning_scheduler(self):
        sparsity = 0.8
        model = nn.Linear(10, 10)
        config = OneShotPruningConfig(
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={
                "initial_sparsity": sparsity,
                "target_sparsity": sparsity,
            },
        )
        scheduler = OneShotPruningScheduler(model, config)
        self.assertAlmostEqual(
            get_tensor_sparsity_ratio(model.weight), sparsity)

    def test_not_to_prune(self):
        class LinearModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = LinearModule()
                self.m2 = LinearModule()

        model = Model()
        scheduler_config = OneShotPruningConfig(
            pruning_fn="unstructured_magnitude", not_to_prune=["m1"])
        scheduler = OneShotPruningScheduler(model, scheduler_config)
        for n, m in model.named_modules():
            if n != 'm2.linear':
                self.assertFalse(hasattr(m, 'get_pruning_parameters'))
        self.assertTrue(hasattr(model.m2.linear, 'get_pruning_parameters'))
        scheduler.remove_pruning()

        scheduler_config = OneShotPruningConfig(
            pruning_fn="unstructured_magnitude", not_to_prune=["linear"])
        scheduler = OneShotPruningScheduler(model, scheduler_config)
        for m in model.modules():
            self.assertFalse(hasattr(m, 'get_pruning_parameters'))
        scheduler.remove_pruning()

    def test_weight_sparsity_map(self):
        class LinearModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1000, 1000)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = LinearModule()
                self.m2 = LinearModule()
                self.m3 = LinearModule()

        m1 = 0.8
        m3 = 0.
        target = 0.4
        model = Model()
        config = OneShotPruningConfig(
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={"initial_sparsity": target},
            weight_sparsity_map={
                "m1": {"initial_sparsity": m1},
                "m3": {"initial_sparsity": m3},
            }
        )
        _ = OneShotPruningScheduler(model, config)
        m1_sparsity = get_tensor_sparsity_ratio(model.m1.linear.weight)
        m2_sparsity = get_tensor_sparsity_ratio(model.m2.linear.weight)
        m3_sparsity = get_tensor_sparsity_ratio(model.m3.linear.weight)
        self.assertAlmostEqual(m1_sparsity, m1, 3)
        self.assertAlmostEqual(m2_sparsity, target, 3)
        self.assertAlmostEqual(m3_sparsity, m3, 3)

    def test_explicit_pruning(self):
        class CustomLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_weight = nn.Parameter(torch.rand(1000, 1000))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_module1 = CustomLayer()
                self.c_module2 = CustomLayer()
                self.linear = nn.Linear(1000, 1000)

        c1 = 0.8
        target = 0.4
        model = Model()
        config = OneShotPruningConfig(
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={"initial_sparsity": target},
            explicit_prune={
                "c_module1": {"name": "c_weight", "initial_sparsity": c1},
                "c_module2": {"name": "c_weight"}
            },
        )
        _ = OneShotPruningScheduler(model, config)
        self.assertTrue(hasattr(model.c_module1, 'get_pruning_parameters'))
        self.assertTrue(hasattr(model.c_module2, 'get_pruning_parameters'))
        c1_sparsity = get_tensor_sparsity_ratio(model.c_module1.c_weight)
        c2_sparsity = get_tensor_sparsity_ratio(model.c_module2.c_weight)
        l_sparsity = get_tensor_sparsity_ratio(model.linear.weight)
        self.assertAlmostEqual(c1_sparsity, c1, 3)
        self.assertAlmostEqual(c2_sparsity, target, 3)
        self.assertAlmostEqual(l_sparsity, target, 3)

    def test_pruning_scheduler_with_patten_lock(self):
        model = nn.Linear(10, 10)
        mask = torch.rand_like(model.weight).le(0.5)
        model.weight.data *= mask
        config = OneShotPruningConfig(pruning_fn="pattern_lock")
        _ = OneShotPruningScheduler(model, config)
        self.assertTrue(model.get_pruning_parameters('mask').eq(mask).all())


if __name__ == '__main__':
    unittest.main()
