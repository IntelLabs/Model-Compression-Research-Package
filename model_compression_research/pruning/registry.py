# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Registry module of pruning methods and schedulers
"""

SCHEDULERS = {}


def register_scheduler(name):
    """Register a scheduler to the registry"""
    def register(scheduler):
        SCHEDULERS[name] = scheduler
        return scheduler
    return register


def register_method(*schedulers, name=None):
    """Register a method to a registered scheduler"""
    def register(method_cls):
        if name is None:
            raise AttributeError(
                f"Must give method {method_cls.__name__} a name when registering")
        for sched_name in schedulers:
            SCHEDULERS[sched_name].PRUNING_FN_DICT[name] = method_cls
        return method_cls
    return register


def get_scheduler_class(name):
    """Get registered scheduler class"""
    return SCHEDULERS[name]


def get_config_class(name):
    """Get registered scheduler matching configuration object"""
    return get_scheduler_class(name).SCHEDULER_CONFIG


def list_schedulers():
    """List all registered schedulers"""
    return list(SCHEDULERS.keys())


def list_methods():
    """List all registered pruning methods"""
    s = set()
    for scheduler in SCHEDULERS.values():
        s.update(scheduler.PRUNING_FN_DICT)
    return list(s)
