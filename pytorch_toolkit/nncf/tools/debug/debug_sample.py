"""
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

import os
import re
from pathlib import Path

import torch


def create_experiment_dir(dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    next_id = 0
    if not re.match(r'\d+', Path(dump_dir).parts[-1]):
        ids = [int(f) for f in os.listdir(dump_dir) if f.isnumeric()]
        if ids:
            next_id = max(ids) + 1
    dump_path = os.path.join(dump_dir, str(next_id))
    os.makedirs(dump_path)
    return dump_path


def register_dump_hooks(model, dump_dir, num_item_to_dump=10):
    os.makedirs(dump_dir, exist_ok=True)
    next_id = 0
    if not re.match(r'\d+/\d+', Path(dump_dir).parts[-1]):
        ids = [int(f) for f in os.listdir(dump_dir) if f.isnumeric()]
        if ids:
            next_id = max(ids) + 1
    print(next_id)
    dump_path = os.path.join(dump_dir, str(next_id))
    os.makedirs(dump_path)
    handles = []
    name_idx = 0
    for name, module in model.named_modules():
        if name:
            module.full_dump_path = \
                os.path.join(dump_path, str("{:04d}".format(name_idx)) + '_' + name.replace('/', '_'))
            name_idx += 1

            def hook(module, inputs, output):
                idx = 0
                for input_ in inputs:
                    path = module.full_dump_path + '_input_{}'.format(str(idx))
                    # print('saving input to {}'.format(path))
                    data = torch.flatten(input_)[:num_item_to_dump].cpu()
                    torch.save(data, path)
                    idx += 1
                path = module.full_dump_path + '_output'
                # print('saving output to {}'.format(path))
                data = torch.flatten(output)[:num_item_to_dump].cpu()
                torch.save(data, path)
                if hasattr(module, 'weight'):
                    data = torch.flatten(module.weight)[:num_item_to_dump].cpu()
                    path = module.full_dump_path + '_weight'
                    print('saving weight to {}'.format(path))
                    torch.save(data, path)
                if hasattr(module, 'scale'):
                    data = module.scale.cpu()
                    path = module.full_dump_path + '_scale'
                    print('saving scales to {}'.format(path))
                    torch.save(data, path)

            handles.append(module.register_forward_hook(hook))
    return handles
