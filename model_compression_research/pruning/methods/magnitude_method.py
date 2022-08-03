# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Unstructured magnitude based pruning method
"""
from collections import defaultdict
import weakref

import torch
from torch.nn import functional as F

from .method import PruningMethod
from .methods_utils import calc_pruning_threshold, handle_block_pruning_dims
from ..registry import register_method


BLOCK_POOLING_FN_LUT = {'max': F.max_pool2d,
                        'avg': F.avg_pool2d}


@register_method('iterative', 'one_shot', name='unstructured_magnitude')
class UnstructuredMagnitudePruningMethod(PruningMethod):
    """Unstructured magnitude pruning"""

    def _init(self, target_sparsity=0., initial_sparsity=0., threshold_decay=0., fast_threshold=False):
        self.register_name('original')
        self.register_name('mask')
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self._current_sparsity = initial_sparsity
        self.threshold_decay = threshold_decay
        self.fast_threshold = fast_threshold
        self._threshold = 0.
        original = getattr(self.module, self.name)
        delattr(self.module, self.name)
        # register weight and mask to layer with new names
        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(self.get_name('mask'), torch.ones_like(
            original, dtype=original.dtype, device=original.device))

    def _update_threshold(self, tensor):
        "Updates the pruning threshold"
        self._threshold = calc_pruning_threshold(
            tensor, self._current_sparsity, self._threshold, self.threshold_decay, fast=self.fast_threshold)

    @torch.no_grad()
    def _compute_mask(self):
        """Compute mask to zero all low magnitude weights"""
        # calculate new threshold
        original = self.get_parameters('original').abs()
        self._update_threshold(original)
        # compute mask according new threshold
        new_mask = (original > self._threshold).to(original.dtype)
        self.set_parameter('mask', new_mask)

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return original * mask

    def _update_mask(self, sparsity_schedule=None):
        """Updates mask of pruned layer according to new target sparsity if one is provided"""
        if sparsity_schedule is not None:
            self._current_sparsity = self.initial_sparsity + \
                (self.target_sparsity - self.initial_sparsity) * sparsity_schedule

    def extra_repr(self):
        s = super().extra_repr()
        s += ', target_sparsity={}'.format(self.target_sparsity)
        if self.threshold_decay > 0.:
            s += ', threshold_decay={}'.format(self.threshold_decay)
        return s


@register_method('iterative', 'one_shot', name='uniform_magnitude')
class UniformMagnitudePruningMethod(UnstructuredMagnitudePruningMethod):
    """Uniform magnitude pruning"""

    def _init(self, target_sparsity=0., initial_sparsity=0., block_size=1):
        super()._init(target_sparsity=target_sparsity, initial_sparsity=initial_sparsity)
        self.block_size = block_size

    def _update_threshold(self, tensor):
        "Updates the pruning threshold"
        self._threshold = calc_pruning_threshold(
            tensor, self._current_sparsity, block_size=self.block_size).unsqueeze(-1)

    @torch.no_grad()
    def _compute_mask(self):
        """Compute mask to zero all low magnitude weights"""
        # calculate new threshold
        original = self.get_parameters('original').abs()
        shape = original.shape
        self._update_threshold(original)
        # compute mask according new threshold
        new_mask = (original.view(-1, self.block_size) >
                    self._threshold).to(original.dtype).view(shape)
        self.set_parameter('mask', new_mask)


@register_method('iterative', 'one_shot', name='block_structured_magnitude')
class BlockStructuredMagnitudePruningMethod(UnstructuredMagnitudePruningMethod):
    """Block magnitude pruning"""

    def _init(self, target_sparsity=0., initial_sparsity=0., threshold_decay=0., block_dims=1, pooling_type='avg', fast_threshold=False):
        self.block_dims = block_dims
        self.pooling_type = pooling_type
        super()._init(target_sparsity, initial_sparsity, threshold_decay, fast_threshold)
        # Handle block dims
        original = getattr(self.module, self.get_name('original'))
        if original.dim() > 2:
            raise NotImplementedError(
                "Currently works only for 2D weights, {}D weight was given".format(original.dim()))
        self.block_dims, self.expanded_shape, self.pooled_shape = handle_block_pruning_dims(
            block_dims, original.shape)
        # Handle pooling function
        if isinstance(self.pooling_type, str):
            self.pooling_type = BLOCK_POOLING_FN_LUT[self.pooling_type]
        else:
            raise ValueError("Pooling type argument must be either from the list {}, got {}".format(
                list(BLOCK_POOLING_FN_LUT.keys()), self.pooling_type))

    @torch.no_grad()
    def _compute_mask(self):
        """Compute mask to zero all low magnitude weights"""
        # calculate new threshold
        original = self.get_parameters('original').abs()
        # pooling weight
        pooled_weight = self.pooling_type(
            original.unsqueeze(0), self.block_dims, self.block_dims).squeeze()
        self._update_threshold(pooled_weight)
        # compute mask according new threshold and expand
        new_mask = (pooled_weight > self._threshold).to(original.dtype).reshape(self.pooled_shape).expand(
            self.expanded_shape).reshape_as(original)
        self.set_parameter('mask', new_mask)

    def extra_repr(self):
        s = super().extra_repr()
        s += ', block_dims={}, pooling_type={}'.format(
            self.block_dims, self.pooling_type.__name__)
        return s


class UnstructuredSparsityGroup:
    """Group multiple existing pruning methods. Pruned tensors of the same group are treated as if they were concatenated to a single tensor"""

    def __init__(self):
        self.target_sparsity = 0.
        self.initial_sparsity = 0
        self._current_sparsity = self.initial_sparsity
        self.threshold_decay = 0.
        self.threshold = 0.
        self.bins = 1024
        self.epsilon = 1e-3
        self.method_list = []

    def add(
        self,
        method,
        group_target_sparsity=None,
        group_initial_sparsity=None,
        group_threshold_decay=None
    ):
        """Add an existing pruning method to group"""
        self.method_list.append(weakref.ref(method))
        if group_target_sparsity is not None:
            self.target_sparsity = group_target_sparsity
        if group_initial_sparsity is not None:
            self.initial_sparsity = group_initial_sparsity
            self._current_sparsity = self.initial_sparsity
        if self.threshold_decay is not None:
            self.threshold_decay = group_threshold_decay

    def update_sparsity(self, sparsity_schedule=None):
        """Update group sparsity"""
        if sparsity_schedule is not None:
            self._current_sparsity = self.initial_sparsity + \
                (self.target_sparsity - self.initial_sparsity) * sparsity_schedule

    def _compute_cdf(self, maximum):
        """Compute the cdf of the pruned tensors in the group"""
        device = maximum.device
        hist = torch.zeros(self.bins, device=device)
        for m in self.method_list:
            if m() is not None:
                hist += torch.histc(getattr(m().module, m().get_name('original')).abs().flatten(),
                                    bins=self.bins, max=maximum.item())
        return hist.cumsum(0)

    def _comptue_maximum_magnitude(self):
        """Compute the maximum magnitude weight in the group"""
        max_mag = torch.stack([getattr(m().module, m().get_name('original')).abs().max()
                               for m in self.method_list if m() is not None]).max()
        return max_mag

    # TODO(Ofir) check if a hook can be added to avoid doing this calculation multiple times if tensors didn't changed
    def compute_new_threshold(self):
        """Compute the pruning threshold of the group"""
        # get the maximum magnitude of the group
        if self._current_sparsity == 0.:
            self.threshold = 0.
        else:
            max_mag = self._comptue_maximum_magnitude()
            cdf = self._compute_cdf(max_mag)
            n = cdf[-1]
            thresh_bin = torch.sum(cdf / cdf[-1] <= 1.0 - self.epsilon)
            if thresh_bin < self.bins / 2:
                max_mag = thresh_bin.float() / self.bins * max_mag
                cdf = self._compute_cdf(max_mag)
                cdf[-1] = n
            normed_cdf = cdf / n
            new_threshold = torch.sum(
                normed_cdf < self._current_sparsity) * max_mag / self.bins
            self.threshold = self.threshold * self.threshold_decay + \
                (1 - self.threshold_decay) * new_threshold
        return self.threshold


class GroupedUnstructuredMagnitudePruningMethod(PruningMethod):
    """
    Grouped maginitude pruning. This method calculates a global threshold for each group of 
    weights being pruned resulting in un-uniform sparsity in the weights.
    """
    GROUPS = defaultdict(UnstructuredSparsityGroup)

    def _init(
        self,
        group='default',
        group_target_sparsity=0.,
        group_initial_sparsity=0.,
        group_threshold_decay=0.,
    ):
        self.register_name('original')
        self.register_name('mask')
        original = getattr(self.module, self.name)
        delattr(self.module, self.name)
        # register weight and mask to layer with new names
        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(self.get_name('mask'), torch.ones_like(
            original, dtype=original.dtype, device=original.device))
        self.group_name = group
        self.group = self.get_group(self.group_name)
        self.group.add(
            self,
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=group_threshold_decay,
        )

    @torch.no_grad()
    def _compute_mask(self):
        original = self.get_parameters('original').abs()
        new_mask = (original > self.group.compute_new_threshold()).to(
            original.dtype)
        self.set_parameter('mask', new_mask)

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return original * mask

    def extra_repr(self):
        s = super().extra_repr()
        s += ', group="{}", group_target_sparsity={}'.format(
            self.group_name, self.group.target_sparsity)
        if self.group.threshold_decay > 0.:
            s += ', group_threshold_decay={}'.format(
                self.group.threshold_decay)
        return s

    def _update_mask(self, sparsity_schedule=None):
        if sparsity_schedule is not None:
            self.group.update_sparsity(sparsity_schedule)

    @classmethod
    def get_group(cls, group_name):
        """Get a sparsity group by name"""
        return cls.GROUPS[group_name]

    @classmethod
    def update_group_sparsity(cls, group_name, sparsity_schedule=None):
        """Update the target sparsity of a sparsity group"""
        group_o = cls.get_group(group_name)
        if sparsity_schedule is not None:
            group_o.update_sparsity(sparsity_schedule)
        for m in group_o.method_list:
            if m() is not None:
                m().update_mask()


@register_method('iterative', 'one_shot', name='global_unstructured_magnitude')
class GlobalUnstructuredMagnitudePruningMethod(GroupedUnstructuredMagnitudePruningMethod):
    """
    Global maginitude pruning. This method calculates a global threshold for each group of 
    weights being pruned resulting in un-uniform sparsity in the weights.
    """

    def _init(
        self,
        group_target_sparsity=0.,
        group_initial_sparsity=0.,
        group_threshold_decay=0.,
    ):
        super()._init(
            group='global',
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=group_threshold_decay,
        )

    @classmethod
    def update_group_sparsity(cls, sparsity_schedule=None):
        super().update_group_sparsity('global', sparsity_schedule=sparsity_schedule)


def unstructured_magnitude_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., threshold_decay=0., fast_threshold=False):
    """Apply magnitude pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = UnstructuredMagnitudePruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            threshold_decay=threshold_decay,
            fast_threshold=fast_threshold,
        )
    return module, method


def uniform_magnitude_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., block_size=1):
    """Apply magnitude pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = UniformMagnitudePruningMethod(
            module,
            name,
            target_sparsity=target_sparsity,
            initial_sparsity=initial_sparsity,
            block_size=block_size,
        )
    return module, method


def block_structured_magnitude_pruning(module, name='weight', target_sparsity=0., initial_sparsity=0., threshold_decay=0., block_dims=1, pooling_type='avg', fast_threshold=False):
    """Apply block magnitude pruning method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
        method.target_sparsity = target_sparsity
        method.update_mask(1)
    except AttributeError:
        method = BlockStructuredMagnitudePruningMethod(
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


def grouped_unstructured_magnitude_pruning(
    module=None,
    name='weight',
    group='default',
    group_target_sparsity=0.,
    group_initial_sparsity=0.,
    threshold_decay=0.,
):
    """Grouped magnitude pruning and assing to group method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
    except AttributeError:
        method = None
    if module is None or method is not None:
        GroupedUnstructuredMagnitudePruningMethod.get_group(
            group).target_sparsity = group_target_sparsity
        GroupedUnstructuredMagnitudePruningMethod.update_group_sparsity(
            group, 1)
    else:
        method = GroupedUnstructuredMagnitudePruningMethod(
            module,
            name,
            group=group,
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=threshold_decay,
        )
    return module, method


def global_unstructured_magnitude_pruning(
    module=None,
    name='weight',
    group_target_sparsity=0.,
    group_initial_sparsity=0.,
    threshold_decay=0.,
):
    """Global magnitude pruning and assing to group method to module"""
    try:
        method = module.get_pruning_parameters('method', name=name)
    except AttributeError:
        method = None
    if module is None or method is not None:
        GlobalUnstructuredMagnitudePruningMethod.get_group(
            'global').target_sparsity = group_target_sparsity
        GlobalUnstructuredMagnitudePruningMethod.update_group_sparsity(1)
    else:
        method = GlobalUnstructuredMagnitudePruningMethod(
            module,
            name,
            group_target_sparsity=group_target_sparsity,
            group_initial_sparsity=group_initial_sparsity,
            group_threshold_decay=threshold_decay,
        )
    return module, method
