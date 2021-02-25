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
from .nmt_tokenizer import NMTTokenizer
from .spbpe_tokenizer import SPBPETokenizer


TOKENIZERS = {
    "spbpe": SPBPETokenizer
}

def build_tokenizer(cfg):
    src = TOKENIZERS[cfg.src.type](**cfg.src.params)
    tgt = TOKENIZERS[cfg.tgt.type](**cfg.tgt.params)
    return NMTTokenizer(src, tgt)
