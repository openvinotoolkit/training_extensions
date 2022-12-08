# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#

import numpy as np
import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose

from otx.algorithms.common.adapters.mmcv.data_cpu import scatter_cpu


def get_fake_input(cfg, data=None, orig_img_shape=(128, 128, 3), device="cuda"):
    if data is None:
        data = dict(img=np.zeros(orig_img_shape, dtype=np.uint8))
    else:
        data = dict(img=data)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)

    for key, value in data.items():
        if not isinstance(value, list):
            data[key] = [value]

    if device == torch.device("cpu"):
        data = scatter_cpu(collate([data], samples_per_gpu=1))[0]
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device.index])[0]
    return data
