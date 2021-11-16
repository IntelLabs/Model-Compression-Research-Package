# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Apply pattern lock pruning mask on weight
"""
from .custom_method import CustomMaskPruningMethod
from ..registry import register_method


@register_method('one_shot', name='pattern_lock')
class PatternLockPruningMethod(CustomMaskPruningMethod):
    """Pattern lock pruning method. Locks found sparsity patterns in place and allows only unpruned weights to change"""

    def _init(self):
        super()._init(self.get_sparsity_pattern_mask())

    def get_sparsity_pattern_mask(self):
        original = getattr(self.module, self.name)
        return original.ne(0.).to(original.dtype)

    def _update_mask(self):
        return super()._update_mask(self.get_sparsity_pattern_mask())


def lock_tensor_sparsity_pattern(module, name='weight'):
    """Apply pattern lock pruning to module"""
    try:
        method = module.get_pruning_parameters(
            'method', name=name).update_mask()
    except AttributeError:
        method = CustomMaskPruningMethod(module, name)
    return module, method
