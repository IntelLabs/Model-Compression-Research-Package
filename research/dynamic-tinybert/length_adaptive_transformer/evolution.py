# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# 
# This file is copied from https://github.com/clovaai/length-adaptive-transformer

# Length-Adaptive Transformer
# Copyright (c) 2020-present NAVER Corp.
# Apache License v2.0

import csv
import logging
import os
import random

import numpy as np
import torch
import torchprofile

logger = logging.getLogger(__name__)


def approx_ratio(x, n=12, l=384):
    s = 0
    i = l
    for _ in range(n):
        i = int(np.ceil(i * (1 - x)))  # i * x
        s += i
    return s / (n * l)


def inverse(x):
    l, r = 0, 1
    while r - l > 1e-12:
        c = (l + r) / 2
        v = approx_ratio(c)
        l, r = (c, r) if x <= v else (l, c)
    return l


def store2str(gene, macs, score, method, parents=None):
    store_str = f"({', '.join(f'{x:3d}' for x in gene)}):"
    store_str += f" {macs} MACs"
    store_str += f" | score {score}"
    store_str += f" | method {method}"
    if parents is not None:
        store_str += f"| parent(s) {parents}"
    return store_str


class Evolution(object):
    def __init__(
        self,
        model,
        args,
        evaluate,
        tokenizer,
        lower_constraint=0,
        upper_constraint=None
    ):
        self.model = model
        self.args = args
        self.evaluate = evaluate
        self.tokenizer = tokenizer

        size = (1, self.args.max_seq_length)
        self.dummy_inputs = (
            torch.ones(size, dtype=torch.long).to(self.args.device),
            torch.ones(size, dtype=torch.long).to(self.args.device),
            torch.zeros(size, dtype=torch.long).to(self.args.device),
        )
        if self.model.config.model_type == "distilbert":
            self.dummy_inputs = self.dummy_inputs[:2]

        self.lower_constraint = lower_constraint
        self.upper_constraint = upper_constraint

        self.store = {}  # gene: (macs, score, method, parent(s))
        self.population = []

    def load_store(self, store_file):
        if not os.path.isfile(store_file):
            return
        with open(store_file, 'r') as f:
            for row in csv.reader(f, delimiter='\t'):
                row = tuple(eval(x) for x in row[:3])
                self.store[row[0]] = row[1:3] + (0, None)

    def save_store(self, store_file):
        store_keys = sorted(self.store.keys(), key=lambda x: self.store[x][0])
        with open(store_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in store_keys:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def save_population(self, population_file, population):
        with open(population_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            for gene in population:
                writer.writerow([str(gene)] + [str(x) for x in self.store[gene]])

    def ccw(self, gene0, gene1, gene2):
        x0, y0 = self.store[gene0][:2]
        x1, y1 = self.store[gene1][:2]
        x2, y2 = self.store[gene2][:2]
        return (x0 * y1 + x1 * y2 + x2 * y0) - (x0 * y2 + x1 * y0 + x2 * y1)

    def convex_hull(self):
        hull = self.population[:2]
        for gene in self.population[2:]:
            if self.store[hull[-1]][1] >= self.store[gene][1]:
                continue
            while len(hull) >= 2 and self.ccw(hull[-2], hull[-1], gene) >= 0:
                del hull[-1]
            hull.append(gene)
        return hull

    def pareto_frontier(self):
        self.population = sorted(self.population, key=lambda x: self.store[x][:2])

        frontier = [self.population[0]]
        for gene in self.population[1:-1]:
            if self.store[gene][1] > self.store[frontier[-1]][1]:
                if self.store[gene][0] == frontier[-1][0]:
                    del frontier[-1]
                frontier.append(gene)
        frontier.append(self.population[-1])
        self.population = frontier

        area = 0
        for gene0, gene1 in zip(self.population[:-1], self.population[1:]):
            x0, y0 = self.store[gene0][:2]
            x1, y1 = self.store[gene1][:2]
            area += (x1 - x0) * y0
        area /= (self.upper_constraint - self.lower_constraint)
        return self.population, area

    def add_gene(self, gene, macs=None, score=None, method=0, parents=None):
        if gene not in self.store:
            self.model.eval()
            if self.model.config.model_type == "distilbert":
                self.model.distilbert.set_length_config(gene)
            elif self.model.config.model_type == "roberta":
                self.model.roberta.set_length_config(gene)
            else:
                assert hasattr(self.model, "bert")
                self.model.bert.set_length_config(gene)
                
            macs = macs or torchprofile.profile_macs(self.model, args=self.dummy_inputs)
            if macs < self.lower_constraint:
                return False
            score = score or self.evaluate(self.args, self.model, self.tokenizer)[0]['f1']
            self.store[gene] = (macs, score, method, parents)
            logger.info(store2str(gene, macs, score, method, parents))

        macs = self.store[gene][0]
        if macs >= self.lower_constraint \
                and (self.upper_constraint is None or macs <= self.upper_constraint) \
                and gene not in self.population:
            self.population.append(gene)
            return True
        return False

    def mutate(self, mutation_prob):
        gene = random.choice(self.population)
        mutated_gene = ()
        for i in range(self.model.config.num_hidden_layers):
            if np.random.uniform() < mutation_prob:
                prev = (self.args.max_seq_length if i == 0 else mutated_gene[i - 1])
                next = (2 if i == self.model.config.num_hidden_layers - 1 else gene[i + 1])
                mutated_gene += (random.randrange(next, prev + 1),)
            else:
                mutated_gene += (gene[i],)
        return self.add_gene(mutated_gene, method=1, parents=(gene,))

    def crossover(self):
        gene0, gene1 = random.sample(self.population, 2)
        crossovered_gene = tuple((g0 + g1 + 1) // 2 for g0, g1 in zip(gene0, gene1))
        return self.add_gene(crossovered_gene, method=2, parents=(gene0, gene1))
