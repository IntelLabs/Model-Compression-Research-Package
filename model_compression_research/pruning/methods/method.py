# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pruning method base class
PruningMethod implementation was inpired by https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/prune.py
"""
import abc
from functools import wraps
import logging

from torch import nn


logger = logging.getLogger(__name__)


class PruningMethod(abc.ABC):
    """Base class for pruning method"""

    def __init__(self, module, name='weight', **kwargs):
        """Initialize pruning method for a specific module"""
        self.module = module
        if not hasattr(module, name):
            raise AttributeError(
                f"Module of type: {type(module).__name__} doesn't have attribute with name: {name}")
        self.name = name
        self.names_dict = {}
        self.register_name('method')
        if hasattr(module, self.get_name('method')):
            raise RuntimeError(
                f"Module's parameter named: {name} already has {type(getattr(module, self.get_name('method'))).__name__} applied to it")
        self._init(**kwargs)
        self.set_parameter('method', self)

        def get_pruning_parameters(module, *params, name='weight'):
            method = getattr(module, self._generate_name(name, 'method'))
            return method.get_parameters(*params, module=module)
        self.module.get_pruning_parameters = get_pruning_parameters.__get__(
            self.module)
        self.compute_mask()
        # recalculate masked weight before each forward step
        self.remove_handler = self.module.register_forward_pre_hook(self)
        # add pruning method to the module's repr string
        assert isinstance(self.module, nn.Module)
        self._old_extra_repr = self.module.extra_repr

        @wraps(self._old_extra_repr.__func__)
        def extra_repr(module):
            method = getattr(module, self.get_name('method'))
            s = method._old_extra_repr()
            s += f'\n  (pruning): {str(self).replace("PruningMethod", "")}'
            return s
        self.module.extra_repr = extra_repr.__get__(self.module)

    def _init(self):
        """Aux init function. This method will be called before the mask is computed when initializing method object"""

    def __call__(self, module, inputs):
        """apply mask to weights and place in module.weight"""
        # when working with nn.DataParallel it is necessery to use the module given and not the
        # one saved inside the method since those layers are on different devices when executing
        setattr(module, self.name, self.masked_weight(module))

    def compute_mask(self):
        """Call subclass compute mask method and apply it to the target tensor"""
        self._compute_mask()
        setattr(self.module, self.name, self.masked_weight(self.module))

    @abc.abstractmethod
    def _compute_mask(self):
        """Compute sparsity mask, needs to be implemented by subclass"""

    @abc.abstractmethod
    def masked_weight(self, module):
        """Calculate the masked weight, needs to implemented by subclass"""

    def _update_mask(self, *args, **kwargs):
        """To be overrided by subclass. called before compute_mask method is called in update_mask method"""

    def update_mask(self, *args, **kwargs):
        """Recompute the sparsity mask according to update arguments"""
        self._update_mask(*args, **kwargs)
        self.compute_mask()

    def remove(self):
        """remove added parameters to module and embed mask to weight"""
        masked_weight = self.masked_weight(self.module)
        for name in self.names_dict.values():
            delattr(self.module, name)
        delattr(self.module, 'get_pruning_parameters')
        self.remove_handler.remove()
        self.module.extra_repr = self._old_extra_repr
        delattr(self.module, self.name)
        self.module.register_parameter(self.name, nn.Parameter(masked_weight))

    def extra_repr(self):
        return 'module_type={}, pruned_tensor="{}"'.format(self.module.__class__.__name__, self.name)

    def register_name(self, name):
        """Register an attribute name in the pruned module"""
        if name in self.names_dict:
            logger.warning(
                "Name {} was already registered to {}".format(name, str(self)))
        self.names_dict[name] = self._generate_name(self.name, name)

    @staticmethod
    def _generate_name(weight_name, param_name):
        """Generate attribute name"""
        return f'{weight_name}_{param_name}'

    def get_name(self, name):
        """Get registered attribute name"""
        return self.names_dict[name]

    def get_module(self, module=None):
        """Get method's module in case module is not given"""
        if module is None:
            module = self.module
        return module

    def get_parameters(self, *args, module=None):
        """Get parameters from module by name"""
        module = self.get_module(module)
        params = [getattr(module, self.get_name(name)) for name in args]
        return params if len(params) > 1 else params[0]

    def set_parameter(self, name, value, module=None):
        """Set module parameters by name"""
        module = self.get_module(module)
        setattr(module, self.get_name(name), value)
        return value

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += self.extra_repr()
        s += ')'
        return s
