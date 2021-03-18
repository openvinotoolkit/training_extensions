"""
MIT License

Copyright (c) 2019 luopeixiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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

import json
import pickle as pkl
from os.path import join

import torch

START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3
NUMBER_SIGNS = set('0123456789.')


def split_number(sign):
    if set(sign) <= NUMBER_SIGNS:
        return list(sign)
    return [sign]


class Vocab:
    def __init__(self, loaded_sign2id=None, loaded_id2sign=None):
        if loaded_id2sign is None and loaded_sign2id is None:
            self.sign2id = {'<s>': START_TOKEN, '</s>': END_TOKEN,
                            '<pad>': PAD_TOKEN, '<unk>': UNK_TOKEN}
            self.id2sign = dict((idx, token)
                                for token, idx in self.sign2id.items())
            self.length = 4
        else:
            assert isinstance(loaded_id2sign, dict) and isinstance(loaded_sign2id, dict)
            assert len(loaded_id2sign) == len(loaded_sign2id)
            self.sign2id = loaded_sign2id
            self.id2sign = loaded_id2sign
            self.length = len(loaded_id2sign)

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def add_phrase(self, phrase):
        for sign in phrase:
            self.add_sign(sign)

    def __len__(self):
        return self.length

    def construct_phrase(self, indices, max_len=None, ignore_end_token=False):
        phrase_converted = []
        if max_len is not None:
            indices_to_convert = indices[:max_len]
        else:
            indices_to_convert = indices

        for token in indices_to_convert:
            if isinstance(token, torch.Tensor):
                val = token.item()
            else:
                val = token
            if val in (PAD_TOKEN, END_TOKEN) and not ignore_end_token:
                break
            phrase_converted.append(self.id2sign.get(val, '?'))

        return ' '.join(phrase_converted)


def write_vocab(data_dir, as_json=True, annotation_file='formulas.norm.lst'):
    """
    traverse training phrases to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()
    annotation_path = join(data_dir, annotation_file)
    with open(annotation_path, 'r') as f:
        texts = [text.strip('\n') for text in f]

    train_filter = 'train_filter.lst'
    with open(join(data_dir, train_filter), 'r') as f:

        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            text = texts[idx].split()
            text_splitted_numbers = []
            for sign in text:
                text_splitted_numbers += split_number(sign)

            vocab.add_phrase(text_splitted_numbers)

    vocab_file = join(data_dir, 'vocab.{}'.format('json' if as_json else 'pkl'))
    print('Writing Vocab File in ', vocab_file)

    dict_to_store = {
        'id2sign': vocab.id2sign,
        'sign2id': vocab.sign2id,
    }
    if as_json:
        with open(vocab_file, 'w') as w:
            json.dump(dict_to_store, w, indent=4, sort_keys=True)
    else:
        with open(vocab_file, 'wb') as w:
            pkl.dump(dict_to_store, w)


def read_vocab(vocab_path):
    if vocab_path.endswith('.pkl'):
        with open(vocab_path, 'rb') as f:
            vocab_dict = pkl.load(f)
    elif vocab_path.endswith('.json'):
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
            vocab_dict['id2sign'] = {int(k): v for k, v in vocab_dict['id2sign'].items()}
    else:
        raise ValueError('Wrong extension of the vocab file')
    vocab = Vocab(loaded_id2sign=vocab_dict['id2sign'], loaded_sign2id=vocab_dict['sign2id'])
    return vocab


def pkl_to_json(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab_dict = pkl.load(f)
        dict_to_store = {
            'id2sign': vocab_dict['id2sign'],
            'sign2id': vocab_dict['sign2id'],
        }
        json_path = vocab_path.replace('.pkl', '.json')
        with open(json_path, 'w') as dest:
            json.dump(dict_to_store, dest, indent=4, sort_keys=True)
