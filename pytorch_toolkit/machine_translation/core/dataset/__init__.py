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
from .nmt_dataset import NMTDataset
from .text_container import TextContainer
from .lmdb_container import LMDBContainer


CONTAINERS = {
    "txt": TextContainer,
    "lmdb": LMDBContainer
}

def build_dataset(cfg):
    src = CONTAINERS[cfg.src.type](cfg.src.corpus)
    tgt = CONTAINERS[cfg.tgt.type](cfg.tgt.corpus)
    return NMTDataset(src, tgt)
