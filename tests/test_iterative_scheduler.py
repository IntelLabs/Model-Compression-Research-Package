# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test Iterative Scheduler
"""
import unittest

import torch
from torch import nn

from model_compression_research import (
    get_tensor_sparsity_ratio,
    IterativePruningConfig,
    IterativePruningScheduler,
)


class TestIterativePruningScheduler(unittest.TestCase):
    def test_pruning_scheduler(self):
        sparsity = 0.8
        model = nn.Linear(10, 10)
        config = IterativePruningConfig(
            pruning_frequency=1,
            policy_begin_step=5,
            policy_end_step=10,
            begin_pruning_step=5,
            end_pruning_step=12,
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={
                "target_sparsity": sparsity,
            },
        )
        scheduler = IterativePruningScheduler(model, config)
        self.assertAlmostEqual(get_tensor_sparsity_ratio(model.weight), 0.)
        for _ in range(5):
            self.assertFalse(scheduler._is_pruning_step())
            self.assertAlmostEqual(scheduler.get_sparsity_schedule(), 0.)
            scheduler.step()
            self.assertAlmostEqual(get_tensor_sparsity_ratio(model.weight), 0.)
        for _ in range(5):
            self.assertTrue(scheduler._is_pruning_step())
            sparsity_schedule = scheduler.get_sparsity_schedule()
            scheduler.step()
            self.assertAlmostEqual(get_tensor_sparsity_ratio(
                model.weight), sparsity_schedule * sparsity, 1)
        self.assertAlmostEqual(scheduler.get_sparsity_schedule(), 1.)
        for _ in range(3):
            self.assertTrue(scheduler._is_pruning_step())
            scheduler.step()
        self.assertFalse(scheduler._is_pruning_step())
        self.assertAlmostEqual(scheduler.get_sparsity_schedule(), 1.)

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
        scheduler_config = IterativePruningConfig(
            pruning_fn="unstructured_magnitude", not_to_prune=['m1'])
        scheduler = IterativePruningScheduler(model, scheduler_config)
        for n, m in model.named_modules():
            if n != 'm2.linear':
                self.assertFalse(hasattr(m, 'get_pruning_parameters'))
        self.assertTrue(hasattr(model.m2.linear, 'get_pruning_parameters'))
        scheduler.remove_pruning()

        scheduler_config = IterativePruningConfig(
            pruning_fn="unstructured_magnitude", not_to_prune=['linear'])
        scheduler = IterativePruningScheduler(model, scheduler_config)
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
        config = IterativePruningConfig(
            pruning_frequency=1,
            policy_begin_step=0,
            policy_end_step=4,
            begin_pruning_step=0,
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={
                "target_sparsity": target,
            },
            weight_sparsity_map={
                "m1": {"target_sparsity": m1},
                "m3": {"target_sparsity": m3},
            }
        )
        scheduler = IterativePruningScheduler(model, config)
        for _ in range(5):
            sparsity_schedule = scheduler.get_sparsity_schedule()
            self.assertTrue(scheduler._is_pruning_step())
            scheduler.step()
            m1_sparsity = get_tensor_sparsity_ratio(model.m1.linear.weight)
            self.assertAlmostEqual(m1_sparsity, sparsity_schedule * m1, 3)
            m2_sparsity = get_tensor_sparsity_ratio(model.m2.linear.weight)
            self.assertAlmostEqual(m2_sparsity, sparsity_schedule * target, 3)
            m3_sparsity = get_tensor_sparsity_ratio(model.m3.linear.weight)
            self.assertAlmostEqual(m3_sparsity, sparsity_schedule * m3, 3)
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
        config = IterativePruningConfig(
            pruning_frequency=1,
            policy_begin_step=0,
            policy_end_step=4,
            begin_pruning_step=0,
            pruning_fn="unstructured_magnitude",
            pruning_fn_default_kwargs={
                "target_sparsity": target,
            },
            explicit_prune={
                "c_module1": {"name": "c_weight", "target_sparsity": c1},
                "c_module2": {"name": "c_weight"},
            },
        )
        scheduler = IterativePruningScheduler(model, config)
        self.assertTrue(hasattr(model.c_module1, 'get_pruning_parameters'))
        self.assertTrue(hasattr(model.c_module2, 'get_pruning_parameters'))
        for _ in range(5):
            sparsity_schedule = scheduler.get_sparsity_schedule()
            self.assertTrue(scheduler._is_pruning_step())
            scheduler.step()
            c1_sparsity = get_tensor_sparsity_ratio(model.c_module1.c_weight)
            self.assertAlmostEqual(c1_sparsity, sparsity_schedule * c1, 3)
            c2_sparsity = get_tensor_sparsity_ratio(model.c_module2.c_weight)
            self.assertAlmostEqual(c2_sparsity, sparsity_schedule * target, 3)
            l_sparsity = get_tensor_sparsity_ratio(model.linear.weight)
            self.assertAlmostEqual(l_sparsity, sparsity_schedule * target, 3)
        self.assertAlmostEqual(c1_sparsity, c1, 3)
        self.assertAlmostEqual(c2_sparsity, target, 3)
        self.assertAlmostEqual(l_sparsity, target, 3)


if __name__ == "__main__":
    unittest.main()
