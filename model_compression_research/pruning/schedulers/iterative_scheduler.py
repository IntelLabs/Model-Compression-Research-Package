# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Iteretive pruning scheduler based on:
To prune, or not to prune: exploring the efficacy of
    pruning for model compression, https://arxiv.org/pdf/1710.01878.pdf
"""

from .scheduler import PruningScheduler, PruningConfig
from ..registry import register_scheduler


class IterativePruningConfig(PruningConfig):
    """Iterative pruning scheduler configuration object

    Arguments: 
        pruning_frequency (uint, optional): (default: 100)
        begin_pruning_step (uint, optional): (default: 0)
        end_pruning_step (int, optional): (default: -1)
        policy_begin_step (uint, optional): (default: 0)
        policy_end_step (uint, optional): (default: 1000)
    """
    ATTRIBUTES = PruningConfig.add_attributes(
        {
            "pruning_frequency": 100,
            "begin_pruning_step": 0,
            "end_pruning_step": -1,
            "policy_begin_step": 0,
            "policy_end_step": 1000,
        }
    )


@register_scheduler('iterative')
class IterativePruningScheduler(PruningScheduler):
    """
    Implements the pruning policy suggested in:
    To prune, or not to prune: exploring the efficacy of
    pruning for model compression, https://arxiv.org/pdf/1710.01878.pdf
    """
    SCHEDULER_CONFIG = IterativePruningConfig
    PRUNING_FN_DICT = {}

    def step(self):
        """Performs a pruning step"""
        if self._is_pruning_step():
            self.prune()
        super().step()

    def tb_log(self):
        if self.writer is not None:
            self.writer.add_scalar(
                'pruning/target_sparsity', self.get_sparsity_schedule(), self.global_step)
            super().tb_log()

    def get_sparsity_schedule(self):
        """
        calculate target sparsity according to formula:
        base = 1 - min{max{[1 - (t - t_0) / (t_1 - t_0)] ^ 3, 0}, 1}
        Pruning method need to adjust the base according to its initial and target sparsity ratios
        s_t = s_i + (s_f - s_i) * base
        """
        t = self.global_step
        t_0 = self.config.policy_begin_step
        t_1 = self.config.policy_end_step
        base = min(max((1 - (t - t_0) / (t_1 - t_0)) ** 3, 0.), 1.)
        return 1 - base

    def _is_pruning_step(self):
        """Checks if next step is a pruning step"""
        to_prune = True
        to_prune &= self.global_step >= self.config.begin_pruning_step
        to_prune &= self.config.end_pruning_step == -1 or \
            self.global_step <= self.config.end_pruning_step
        to_prune &= (self.global_step - self.config.begin_pruning_step) % \
            self.config.pruning_frequency == 0
        return to_prune

    def prune(self):
        """Applies pruning method to the model and all its sublayers"""
        sparsity_schedule = self.get_sparsity_schedule()
        for method_dict in self.method_dicts:
            method_dict['method'].update_mask(sparsity_schedule)

    def extra_repr(self):
        s = 'global_step={}, current_sparsity={:.3f}, '.format(
            self.global_step,
            self.get_sparsity_schedule()
        )
        return s + super().extra_repr()
