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
class NMTTokenizer:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def vocab_size(self, mode='tgt'):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.vocab_size()

    def pad_idx(self, mode='tgt'):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.pad_idx()

    def encode_batch(self, batch, mode="tgt", return_tensors='pt'):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.encode_batch(batch, return_tensors=return_tensors)

    def decode_batch(self, batch, mode="tgt", remove_extra=True, postprocess=False):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.decode_batch(batch, remove_extra=remove_extra, postprocess=postprocess)

    def vocab_len(self, mode='tgt'):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.vocab_len()

    def pad_idx(self, mode='tgt'):
        tokenizer = self.tgt if mode == "tgt" else self.src
        return tokenizer.pad_idx()

    def __call__(self, batch):
        src, tgt = [], []
        for i in range(len(batch)):
            src.append(batch[i]["src"])
            tgt.append(batch[i]["tgt"])
        src = self.src.encode_batch(src, return_tensors='pt')
        tgt = self.tgt.encode_batch(tgt, return_tensors='pt')
        return {
            "src": src,
            "tgt": tgt
        }
