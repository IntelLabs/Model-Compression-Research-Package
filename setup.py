#!/usr/bin/env python
# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import find_packages, setup

root = os.path.abspath(os.path.dirname(__file__))

with open('requirements.txt') as fp:
    install_requirements = fp.readlines()

# read official README.md
with open('README.md', encoding='utf8') as fp:
    long_desc = fp.read()

version = "0.1.0"

setup(name='model_compression_research',
      version=version,
      description='Model Compression Research Package',
      long_description=long_desc,
      long_description_content_type='text/markdown',
      license="Apache",
      author='Ofir Zafrir',
      author_email='ofir.zafrir@intel.com',
      python_requires='>=3.6.*',
      packages=find_packages(
          exclude=['tests.*', 'tests', 'examples', 'examples.*', 'research', 'research.*']),
      install_requires=install_requirements,
      include_package_data=True,
      )
