# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import torch


class GreedyDecoder:
    """Greedy CTC decoder."""
    def __init__(self, tokenizer, blank_id):
        self.tokenizer = tokenizer
        self.blank_id = blank_id

    def decode(self, probs, sizes=None, remove_repetitions=False):
        ids = probs.softmax(-1).argmax(-1)
        return self.convert_to_strings(ids, sizes, remove_repetitions)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False):
        strings = []
        for i in range(len(sequences)):
            tokens = self.process_ctc(
                sequences[i].detach().cpu().tolist(),
                sizes[i] if sizes is not None else len(sequences[i]),
                remove_repetitions
            )
            strings.append(
                self.tokenizer.decode(torch.LongTensor(tokens))[0]
            )
        return strings

    def process_ctc(self, sequence, size, remove_repetitions=False):
        tokens = []
        for i in range(size):
            if sequence[i] != self.blank_id:
                if remove_repetitions and i != 0 and sequence[i] == sequence[i - 1]:
                    pass
                else:
                    tokens.append(sequence[i])
        if not len(tokens):
            tokens.append([])
        return tokens
