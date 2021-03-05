"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sacrebleu
from .utils import all_gather


class BLEU:
    def __init__(self, num_refs=1, distributed=False):
        self.num_refs = num_refs
        self.distributed = distributed
        self.reset()

    def add(self, preds, gt):
        assert len(gt) == self.num_refs
        self._add_to_buf(preds, self.preds)
        for i in range(len(gt)):
            self._add_to_buf(gt[i], self.refs[i])

    def value(self):
        if self.distributed:
            self._reduce()
        metric = {
            'bleu': sacrebleu.corpus_bleu(self.preds, self.refs).score
        }
        return metric

    def reset(self):
        self.preds = []
        self.refs = [[] for i in range(self.num_refs)]

    def _add_to_buf(self, sample, buf):
        if isinstance(sample, list):
            buf.extend(sample)
        elif isinstance(sample, str):
            buf.append(sample)
        else:
            raise 'unvalid text type'

    def _reduce(self):
        preds = all_gather(self.preds)
        refs = all_gather(self.refs)
        self.reset()
        for i in range(len(preds)):
            self.preds.extend(preds[i])
            for j in range(self.num_refs):
                self.refs[j].extend(refs[i][j])
