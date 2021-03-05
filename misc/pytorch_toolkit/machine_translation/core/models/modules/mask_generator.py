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
import torch
import torch.nn as nn

class MaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("device_info", torch.ones(1))

    def forward(self, src_mask, delta, max_len=None):
        src_lens = src_mask.sum(1).long()
        tgt_lens = src_lens + delta
        max_len = tgt_lens.max().item() if max_len is None else max_len
        tgt_lens = torch.clamp(tgt_lens, min=1, max=max_len)
        arange = torch.arange(max_len).to(self.device_info.device)
        tgt_mask = (arange[None, :].repeat(src_mask.size(0), 1) < tgt_lens[:, None]).float().detach()
        return tgt_lens, tgt_mask
