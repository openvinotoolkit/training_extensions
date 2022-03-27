# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Dict, List
import editdistance


class MetricAggregator:
    """Base class for metrics aggregation."""
    def __init__(self, metrics: List = []):
        self.metrics = metrics

    def update(self, pred: str, gt: str) -> Dict[str, float]:
        out = {}
        for m in self.metrics:
            out[m.get_name()] = m.update(pred, gt)
        return out

    def compute(self) -> Dict[str, float]:
        out = {}
        for m in self.metrics:
            out[m.get_name()] = m.compute()
        return out

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()


class CharacterErrorRate:
    """Implementation of Character Error Rate metric for Speech To Text task."""
    def __init__(self):
        super().__init__()
        self.reset()

    def get_name(self) -> str:
        return "cer"

    def update(self, pred: str, gt: str) -> float:
        cur_score = editdistance.eval(gt, pred)
        cur_length = len(gt)
        self.score += cur_score
        self.length += cur_length
        return cur_score / cur_length if cur_length != 0 else 1.0

    def compute(self) -> float:
        return self.score / self.length if self.length != 0 else 1.0

    def reset(self) -> None:
        self.length, self.score = 0, 0


class WordErrorRate:
    """Implementation of Word Error Rate metric for Speech To Text task."""
    def __init__(self):
        super().__init__()
        self.reset()

    def get_name(self) -> str:
        return "wer"

    def update(self, pred: str, gt: str) -> float:
        cur_score = editdistance.eval(gt.split(), pred.split())
        cur_words = len(gt.split())
        self.score += cur_score
        self.words += cur_words
        return cur_score / cur_words if cur_words != 0 else 1.0

    def compute(self) -> float:
        return self.score / self.words if self.words != 0 else 1.0

    def reset(self) -> None:
        self.words, self.score = 0, 0
