# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Discovering neural wiring method 
based on https://arxiv.org/abs/1906.00586
"""

from .magnitude_method import (
    UnstructuredMagnitudePruningMethod,
    BlockStructuredMagnitudePruningMethod,
    UniformMagnitudePruningMethod,
)
from .methods_utils import MaskFilledSTE
from ..registry import register_method


@register_method('iterative', name='unstructured_neural_wiring')
class UnstructuredNeuralWiringPruningMethod(UnstructuredMagnitudePruningMethod):
    """Unstructured magnitude pruning with straight through estimation of gradients of pruned weights"""

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return MaskFilledSTE.apply(original, mask)


@register_method('iterative', name='block_structured_neural_wiring')
class BlockStructuredNeuralWiringPruningMethod(BlockStructuredMagnitudePruningMethod):
    """Block magnitude pruning with straight through estimation of gradients of pruned weights"""

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return MaskFilledSTE.apply(original, mask)


@register_method('iterative', name='uniform_neural_wiring')
class UniformNeuralWiringPruningMethod(UniformMagnitudePruningMethod):
    """Uniform magnitude pruning with straight through estimation of gradients of pruned weights"""

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return MaskFilledSTE.apply(original, mask)


def unstructured_neural_wiring_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., threshold_decay=0., fast_threshold=False):
    """Apply neural wiring pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = UnstructuredNeuralWiringPruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            threshold_decay=threshold_decay,
            fast_threshold=fast_threshold,
        )
    return module, method


def block_structured_neural_wiring_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., threshold_decay=0., block_dims=1, pooling_type='avg', fast_threshold=False):
    """Apply block neural wiring pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = BlockStructuredNeuralWiringPruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            threshold_decay=threshold_decay,
            block_dims=block_dims,
            pooling_type=pooling_type,
            fast_threshold=fast_threshold,
        )
    return module, method


def uniform_neural_wiring_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., block_size=1):
    """Apply neural wiring pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = UniformNeuralWiringPruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            block_size=block_size
        )
    return module, method
