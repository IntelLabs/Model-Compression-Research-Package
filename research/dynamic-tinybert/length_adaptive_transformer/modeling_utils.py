# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# 
# This file is copied from https://github.com/clovaai/length-adaptive-transformer

# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0


def expand_gather(input, dim, index):
    size = list(input.size())
    size[dim] = -1
    return input.gather(dim, index.expand(*size))
