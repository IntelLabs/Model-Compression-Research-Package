#!/bin/bash
# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Use this script to run all tests

GPU=

while test $# -gt 0; do
    case "$1" in
        --gpu)
            shift
            GPU=$1
            shift
            ;;
        *)
            TAIL+=' '$1
            shift
            ;;
    esac
done

CUDA_VISIBLE_DEVICES=$GPU python -m unittest tests/*.py $TAIL