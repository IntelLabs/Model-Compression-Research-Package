# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for pruning methods
"""
from collections.abc import Iterable
from itertools import chain
import warnings

import torch


class MaskFilledSTE(torch.autograd.Function):
    """Mask filling op with estimated gradients using STE"""

    @staticmethod
    def forward(ctx, input, mask):
        """"""
        return input * mask

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-Through Estimator (STE) according to"""
        return grad_output, None


def _calc_pruning_threshold_cdf(tensor, target_sparsity, bins=1024, epsilon=1e-3):
    def _calc_cdf(tensor, max_mag):
        return torch.histc(tensor, bins=bins, max=max_mag.item()).cumsum(0)
    max_mag = tensor.max()
    cdf = _calc_cdf(tensor, max_mag)
    n = cdf[-1]
    thresh_bin = torch.sum(cdf / cdf[-1] <= 1. - epsilon)
    if thresh_bin < bins / 2:
        max_mag = thresh_bin.float() / bins * max_mag
        cdf = _calc_cdf(tensor, max_mag)
        cdf[-1] = n
    normed_cdf = cdf / n
    return torch.sum(normed_cdf < target_sparsity) * max_mag / bins


def calc_pruning_threshold(tensor, target_sparsity, current_threshold=0., threshold_decay=0., block_size=1, fast=False):
    """Calculate a new pruning threhsold for pruning tensor to target sparsity"""
    if tensor.dim() > 2 and block_size != 1:
        raise NotImplementedError(
            "calc_pruning_threshold not implemented yet for 3D tensors and above with block_size > 1, got {}".format(block_size))
    if fast:
        if tensor.lt(0).any():
            warnings.warn(
                "Fast pruning threshold calculation for tensors with negative values not implemented yet. Fall back to regular threshold calculation.")
            fast = False
        if block_size != 1:
            warnings.warn(
                "Fast pruning threshold calculation with `block_size`!=1 not implemented yet. Fall back to regular threshold calculation.")
            fast = False
    if block_size == 1:
        reshaped_tensor = tensor.flatten()
    else:
        reshaped_tensor = tensor.view(-1, block_size)
    k = int(target_sparsity * reshaped_tensor.shape[-1])
    assert k >= 0
    # if k is 0 than we will get 100% sparsity in case of sort
    # or RuntimeError in case of kthvalue
    if k > 0:
        # On gpu sort is substantialy faster than kthvalue
        if fast:
            threshold = _calc_pruning_threshold_cdf(
                reshaped_tensor, target_sparsity)
        if reshaped_tensor.is_cuda:
            threshold = reshaped_tensor.sort().values.t()[k - 1]
        else:
            threshold = reshaped_tensor.kthvalue(k, dim=-1).values
    else:
        threshold = tensor.min() - 1
    threshold = current_threshold * threshold_decay + \
        (1 - threshold_decay) * threshold
    return threshold


def handle_block_pruning_dims(block_dims, tensor_shape):
    tensor_dims = len(tensor_shape)
    if not isinstance(block_dims, Iterable):
        block_dims = tensor_dims * (block_dims, )
    block_dims = tuple(block_dims)
    if tensor_dims < len(block_dims):
        raise ValueError("Block number of dimensions {} can't be higher than the number of the weight's dimension {}".format(
            len(block_dims), tensor_dims))
    if len(block_dims) < tensor_dims:
        # Extend block dimensions with ones to match the number of dimensions of the pruned tensor
        block_dims = (
            tensor_dims - len(block_dims)) * (1, ) + block_dims
    # pytorch transposes the input and output channels
    block_dims = (
        block_dims[1], block_dims[0]) + block_dims[2:]
    expanded_shape = tuple(chain.from_iterable(
        [[d // b, b] for d, b in zip(tensor_shape, block_dims)]))
    pooled_shape = tuple(chain.from_iterable(
        [[d // b, 1] for d, b in zip(tensor_shape, block_dims)]))
    return block_dims, expanded_shape, pooled_shape
