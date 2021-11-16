# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
One shot pruning scheduler, this scheduler will apply the pruning method once 
during initialization and doesn't do any steps. How sparsity developes during training is method dependent
"""
import logging

from ..registry import register_scheduler
from .scheduler import PruningScheduler, PruningConfig


logger = logging.getLogger(__name__)


class OneShotPruningConfig(PruningConfig):
    """Iterative pruning scheduler configuration object"""


@register_scheduler('one_shot')
class OneShotPruningScheduler(PruningScheduler):
    """
    Applies pruning method once during initialization. 
    Sparsity will develop during training according to the method chosen
    """
    SCHEDULER_CONFIG = OneShotPruningConfig
    PRUNING_FN_DICT = {}

    def step(self):
        """This scheduler doesn't perform steps"""
        super().step()
