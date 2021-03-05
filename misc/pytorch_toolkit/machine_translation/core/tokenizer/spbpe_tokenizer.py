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
import os
import torch
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing


class SPBPETokenizer:
    def __init__(self, path, enable_truncation=True, enable_padding=True, max_length=150):
        self.tokenizer = self.load_tokenizer(
            path=path,
            enable_truncation=enable_truncation,
            enable_padding=enable_padding,
            max_length=max_length)
        self.max_length = max_length

    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def pad_idx(self):
        return self.tokenizer.token_to_id('<pad>')

    def encode(self, line):
        tokens = self.tokenizer.encode(
            line,
            add_special_tokens=True
        )
        return tokens.ids

    def encode_batch(self, text, return_tensors='pt'):
        batch = self.tokenizer.encode_batch(text)
        ids = []
        for i in range(len(batch)):
            ids.append(batch[i].ids)
        if return_tensors=='pt':
            ids = torch.LongTensor(ids)
        return ids

    def decode(self, input_ids, remove_extra=True, postprocess=False):
        line = self.tokenizer.decode(
            input_ids,
            skip_special_tokens=True
        )
        if remove_extra:
            line = line.replace('<s> ', '').replace('</s>', '').replace('<pad>', '').lstrip(' ')
        if postprocess:
            line = self._postprocess(line)
        return line

    def decode_batch(self, batch, remove_extra=True, postprocess=False):
        text = []
        for i in range(batch.size(0)):
            text.append(self.decode(batch[i].tolist(), remove_extra, postprocess))
        return text

    def _postprocess(self, line):
        tokens = line.lower().split()
        tokens_new = []
        if len(tokens):
            tokens_new = [tokens[0]]
            for i in range(1, len(tokens)):
                if tokens[i] != tokens_new[-1]:
                    tokens_new.append(tokens[i])
        return " ".join(tokens_new)

    @staticmethod
    def load_tokenizer(path, enable_truncation=True, enable_padding=True, max_length=512):
        tokenizer = SentencePieceBPETokenizer(
            os.path.join(path, "vocab.json"),
            os.path.join(path, "merges.txt")
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        if enable_truncation:
            tokenizer.enable_truncation(max_length=max_length)
        if enable_padding:
            tokenizer.enable_padding(
                pad_token="<pad>",
                pad_id=tokenizer.token_to_id("<pad>"))
        return tokenizer

    @staticmethod
    def train(corpus_list, vocab_size, output, output_name=None):
        print("create tokenizer...")
        tokenizer = SentencePieceBPETokenizer()
        print("load corpus list...")
        corpus_list = open(corpus_list).read().split('\n')[:-1]
        print("train tokenizer...")
        tokenizer.train(
            corpus_list,
            vocab_size=vocab_size,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>"
            ]
        )
        print("save model...")
        tokenizer.save_model(output, output_name)
