# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Apply custom pruning mask on weight
"""
import logging

from .method import PruningMethod


logger = logging.getLogger(__name__)


class CustomMaskPruningMethod(PruningMethod):
    """Prune the target tensor using a given mask"""

    def _init(self, init_mask):
        self.register_name('original')
        self.register_name('mask')
        original = getattr(self.module, self.name)
        delattr(self.module, self.name)
        self.module.register_parameter(self.get_name('original'), original)
        self.module.register_buffer(self.get_name('mask'), None)
        self._set_mask(init_mask)

    def _set_mask(self, mask=None):
        """Since a custom is used there is only need to check that mask is valid"""
        if mask is not None:
            original = self.get_parameters('original')
            if mask.dtype != original.dtype:
                logger.warning("Mask provided data dtype is {}, casting to {}".format(
                    mask.dtype, original.dtype))
                mask = mask.to(original.dtype)
            self.set_parameter('mask', mask.to(original.device))

    def _compute_mask(self):
        return super()._compute_mask()

    def masked_weight(self, module):
        original, mask = self.get_parameters('original', 'mask', module=module)
        return original * mask

    def _update_mask(self, mask=None):
        if mask is not None:
            self._set_mask(mask)


def custom_mask_pruning(module, name='weight', mask=None):
    """Apply custom mask pruning to module. in case init_mask is not supplied the module will not be pruned"""
    try:
        module.get_pruning_parameters('method', name=name).update_mask(mask)
    except AttributeError:
        method = CustomMaskPruningMethod(module, name, init_mask=mask)
    return module, method
