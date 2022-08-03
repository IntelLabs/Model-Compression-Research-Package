# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Testing utils
"""
from absl.testing import parameterized

import torch

from model_compression_research.pruning.methods.methods_utils import (
    MaskFilledSTE,
    calc_pruning_threshold,
)


class TestMaskFilledSTE(parameterized.TestCase):
    def test_forward(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        tensor = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]], device=device)
        mask = torch.tensor([[0, 0, 1],
                             [1, 0, 1],
                             [1, 1, 0]], dtype=torch.bool, device=device)
        ground = torch.masked_fill(tensor, ~mask, 0.)
        out = MaskFilledSTE.apply(tensor, mask)
        self.assertTrue((ground == out).all())

    def test_backward(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        weight = torch.tensor([[1., 2., 3.],
                               [4., 5., 6.],
                               [7., 8., 9.]], requires_grad=True, device=device)
        mask = torch.tensor([[0, 0, 1],
                             [1, 0, 1],
                             [1, 1, 0]], dtype=torch.bool, device=device)
        i = torch.randint(1, 5, (2, 3), dtype=torch.float,
                          requires_grad=True, device=device)
        masked_weight = MaskFilledSTE.apply(weight, mask)
        l = torch.mm(i, masked_weight).sum()
        l.backward()
        self.assertTrue((weight.grad == i.sum(
            0, keepdim=True).t().expand_as(weight)).all())
        self.assertTrue((i.grad == weight.masked_fill(
            ~mask, 0.).sum(1, keepdim=True).t().expand_as(i)).all())


class TestCalcPruningThreshold(parameterized.TestCase):
    def setUp(self):
        self.tensor = torch.tensor([
            [11, 17, 53, 54],
            [15, 26, 37, 88],
            [24, 33, 82, 21],
            [48, 37, 26, 85],
        ]).float()

    @parameterized.parameters([False, True])
    def test_simple_case(self, fast):
        thresh = calc_pruning_threshold(self.tensor, 0.75, fast=fast)
        self.assertEqual(thresh, 53)
        thresh = calc_pruning_threshold(self.tensor, 0., fast=fast)
        self.assertEqual(thresh, 10.)
        thresh = calc_pruning_threshold(self.tensor, 0.75, 22, 0.5, fast=fast)
        self.assertEqual(thresh, 53 * 0.5 + 22 * 0.5)

    def test_block_case(self):
        thresh = calc_pruning_threshold(self.tensor, 0.5, block_size=2)
        ground_truth = torch.tensor([11, 53, 15, 37, 24, 21, 37, 26])
        self.assertTrue((thresh == ground_truth).all())
        thresh = calc_pruning_threshold(self.tensor, 0.5, block_size=4)
        ground_truth = torch.tensor([17, 26, 24, 37])
        self.assertTrue((thresh == ground_truth).all())
        thresh = calc_pruning_threshold(self.tensor, 0.75, block_size=4)
        ground_truth = torch.tensor([53, 37, 33, 48])
        self.assertTrue((thresh == ground_truth).all())
        thresh = calc_pruning_threshold(self.tensor, 0., block_size=4)
        self.assertTrue(thresh == 10.)
        current = torch.tensor([5, 10, 13, 18])
        thresh = calc_pruning_threshold(
            self.tensor, 0.75, current_threshold=current, threshold_decay=0.6, block_size=4)
        ground_truth = ground_truth * 0.4 + current * 0.6
        self.assertTrue((thresh == ground_truth).all())

    def test_nd_tensor_without_blocks(self):
        tensor = self.tensor.view(2, 2, 2, 2)
        self.assertEqual(len(tensor.size()), 4)
        thresh = calc_pruning_threshold(tensor, 0.75)
        self.assertEqual(thresh, 53)
        thresh = calc_pruning_threshold(self.tensor, 0.)
        self.assertEqual(thresh, 10.)
        thresh = calc_pruning_threshold(self.tensor, 0.75, 22, 0.5)
        self.assertEqual(thresh, 53 * 0.5 + 22 * 0.5)

    def test_nd_tensor_with_blocks(self):
        tensor = self.tensor.view(2, 2, 2, 2)
        self.assertEqual(len(tensor.size()), 4)
        with self.assertRaises(NotImplementedError):
            calc_pruning_threshold(tensor, 0.5, block_size=2)


if __name__ == '__main__':
    parameterized.unittest.main()
