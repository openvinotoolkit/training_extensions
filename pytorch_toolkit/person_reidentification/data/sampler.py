"""
 MIT License

 Copyright (c) 2018 Kaiyang Zhou

 Copyright (c) 2019 Intel Corporation

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

from __future__ import absolute_import
from __future__ import division

import numpy as np
import random

from torch.utils.data.sampler import RandomSampler

from torchreid.data.sampler import RandomIdentitySampler


def build_train_sampler(data_source, train_sampler, batch_size=32, num_instances=4, **kwargs):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
    """
    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomIdentitySamplerV2':
        sampler = RandomIdentitySamplerV2(data_source, batch_size, num_instances)
    else:
        sampler = RandomSampler(data_source)

    return sampler


class RandomIdentitySamplerV2(RandomIdentitySampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source, batch_size, num_instances)

    def __iter__(self):
        random.shuffle(self.pids)
        output_ids = []
        for pid in self.pids:
            random.shuffle(self.index_dic[pid])
            output_ids += self.index_dic[pid]
        extra_samples = len(output_ids) % self.num_instances
        output_ids = output_ids[: len(output_ids) - extra_samples]
        ids = np.array(output_ids)
        ids = ids.reshape((-1, self.num_instances))
        np.random.shuffle(ids)
        ids = ids.reshape((-1))
        return iter(ids.tolist())

    def __len__(self):
        return len(self.data_source)
