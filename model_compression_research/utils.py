# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Library utilities
"""
import logging
import copy
import json
import abc
import os


logger = logging.getLogger(__name__)


class Config(abc.ABC):
    """Configuration Object"""
    ATTRIBUTES = {}

    def __init__(self, **kwargs):
        for entry in self.ATTRIBUTES:
            setattr(self, entry, kwargs.pop(entry, self.ATTRIBUTES[entry]))
        if kwargs:
            raise TypeError(
                f"got an unexpected keyword argument: {list(kwargs.keys())}")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a config from a Python dictionary of parameters."""
        config = cls()
        config.update_from_dict(json_object)
        return config

    def update_from_json_file(self, json_file):
        """Update config from json file"""
        self.update_from_dict(self._load_json_file(json_file))

    def update_from_json_string(self, json_string):
        """Update config from json string"""
        if json_string:
            self.update_from_dict(json.loads(json_string))

    def update(self, *args, **kwargs):
        """Update config from either json file, string or dictionary"""
        for arg in args:
            if isinstance(arg, str):
                if os.path.exists(arg):
                    self.update_from_json_file(arg)
                else:
                    self.update_from_json_string(arg)
            elif isinstance(arg, dict):
                self.update_from_dict(arg)
        self.update_from_dict(kwargs)

    @staticmethod
    def _load_json_file(json_file):
        """Load json file and return dictionary representing it"""
        with open(json_file, "r", encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def add_attributes(cls, attributes):
        """Add attributes to config class without overriding existing ones"""
        new_attributes = copy.deepcopy(cls.ATTRIBUTES)
        new_attributes.update(attributes)
        return new_attributes

    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def update_from_dict(self, d):
        """Update config from dictionary"""
        ignored_keys = []
        update_d = {}
        for k in d:
            if k in self.ATTRIBUTES:
                update_d[k] = d[k]
            else:
                ignored_keys.append(k)
        if len(ignored_keys) > 0:
            logger.warning(
                f"Ignored keys when updating config: {ignored_keys}")
        self.__dict__.update(update_d)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs Config from a json file of parameters."""
        return cls.from_dict(cls._load_json_file(json_file))
