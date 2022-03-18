# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Based on Glow-TTS https://github.com/jaywalnut310/glow-tts (with modifications):
# Copyright (c) 2020 Jaehyeon Kim
# SPDX-License-Identifier: MIT
#
# from https://github.com/keithito/tacotron
# Copyright (c) 2017 Keith Ito
# SPDX-License-Identifier: MIT
#
import re
from . import cleaners
from .symbols import symbols
import random
import numpy as np


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence_(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


def get_arpabet(word, dictionary):
  word_arpabet = dictionary.lookup(word)
  if word_arpabet is not None:
    return "{" + word_arpabet[0] + "}"
  else:
    return word

def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  space = _symbols_to_sequence(' ')
  text = text.lstrip()
  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      clean_text = _clean_text(text, cleaner_names)
      if dictionary is not None:
        signs = ".,:'?!\""
        words = [w for w in clean_text.split(" ")]
        tmp = words
        #tmp = []
        # for w in words:
        #   #print("word : ", w)
        #   if w[-1] in signs:
        #     #print(w[:-1], "  ", w[-1])
        #     tmp.append(w[:-1])
        #     tmp.append(w[-1])
        #   else:
        #     tmp.append(w)
        clean_text = [get_arpabet(w, dictionary) for w in tmp]
        for i in range(len(clean_text)):
          t = clean_text[i]
          if t.startswith("{"):
            sequence += _arpabet_to_sequence(t[1:-1])
          else:
            sequence += _symbols_to_sequence(t)
          sequence += space
      else:
        sequence += _symbols_to_sequence(clean_text)
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # remove trailing space
  if dictionary is not None:
    sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'


def intersperse(lst):
  item = len(symbols)
  
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def pad_spaces(seq, sz):
  if type(seq) is list:
    if len(seq) >= sz:
      return seq
    return seq + [_symbol_to_id[' ']] * (sz - len(seq))
  else:
    add_sz = sz - seq.shape[-1]
    pad_width = [(0, 0) for i in range(len(seq.shape) - 1)]
    pad_width += [(0, add_sz)]
    seq = np.pad(seq, pad_width=pad_width, constant_values=_symbol_to_id[' '])
    return seq


def random_fill(lst, p=0.1):
  item = len(symbols) + 1
  trials = min(max(1, int(len(lst) * p)), 30)

  for _ in range(trials):
    idx = random.randint(0, len(lst) - 1)
    lst[idx] = item
  return lst
