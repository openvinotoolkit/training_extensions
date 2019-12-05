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

class AlphabetDecoder:
    """
    This class is used to encode text to numerical sequence and
    to decode it back to symbolic representation.
    """

    NOT_FOUND = -1

    def __init__(self, alphabet=None, make_lower=True):
        if not alphabet:
            alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.sos = "<"
        assert self.sos not in alphabet

        self.eos = '>'
        assert self.eos not in alphabet

        alphabet = self.sos + self.eos + alphabet
        self.alphabet = alphabet
        self.make_lower = make_lower

    def encode(self, input_string):
        """ Encodes text to numerical representation. """

        if not input_string:
            return None

        input_string = input_string.strip()

        if self.make_lower:
            input_string = input_string.lower()

        if self.sos in input_string or self.eos in input_string:
            return None

        res = [self.alphabet.find(character) for character in input_string]

        res.append(self.alphabet.find(self.eos))

        if AlphabetDecoder.NOT_FOUND in res:
            return None

        return res

    def decode(self, numbers):
        """ Decoders numerical representation to text. """
        if numbers is None:
            return None

        output_string = ""
        for number in numbers:
            assert number < len(self.alphabet), "number = {}".format(number)
            if self.alphabet[number] == self.eos:
                break
            output_string += self.alphabet[number]

        return output_string

    def encode_batch(self, strings):
        """ Encodes batch of texts to numerical representations. """
        return [self.encode(string) for string in strings]

    def batch_decode(self, numbers):
        """ Decodes batch of numerical representations to texts. """
        return [self.decode(numbers) for numbers in numbers]

    def string_in_alphabet(self, string):
        """ Returns True if all characters of input string are in alphabet. """
        return self.encode(string) is not None
