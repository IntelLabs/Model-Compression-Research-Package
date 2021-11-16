# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for pruning methods
"""
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


def calc_pruning_threshold(tensor, target_sparsity, current_threshold=0., threshold_decay=0., block_size=1):
    """Calculate a new pruning threhsold for pruning tensor to target sparsity"""
    if tensor.dim() > 2 and block_size != 1:
        raise NotImplementedError(
            "calc_pruning_threshold not implemented yet for 3D tensors and above with block_size > 1, got {}".format(block_size))
    if block_size == 1:
        reshaped_tensor = tensor.flatten()
    else:
        reshaped_tensor = tensor.view(-1, block_size)
    k = int(target_sparsity * reshaped_tensor.shape[-1])
    try:
        threshold = reshaped_tensor.kthvalue(k, dim=-1)[0]
    except RuntimeError:
        threshold = torch.tensor([0.], device=reshaped_tensor.device)
    threshold = current_threshold * threshold_decay + \
        (1 - threshold_decay) * threshold
    return threshold
