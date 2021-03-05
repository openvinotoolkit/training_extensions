# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import print_function
import re
from lpr.trainer import encode, decode_beams

def dataset_size(fname):
  count = 0
  with open(fname, 'r') as file_:
    for _ in file_:
      count += 1
  return count

def lpr_pattern_check(label, lpr_patterns):
  for pattern in lpr_patterns:
    if re.match(pattern, label):
      return True
  return False

def edit_distance(string1, string2):
  len1 = len(string1) + 1
  len2 = len(string2) + 1
  tbl = {}
  for i in range(len1):
    tbl[i, 0] = i
  for j in range(len2):
    tbl[0, j] = j
  for i in range(1, len1):
    for j in range(1, len2):
      cost = 0 if string1[i - 1] == string2[j - 1] else 1
      tbl[i, j] = min(tbl[i, j - 1] + 1, tbl[i - 1, j] + 1, tbl[i - 1, j - 1] + cost)

  return tbl[i, j]


def accuracy(label, val, vocab, r_vocab, lpr_patterns):
  pred = decode_beams(val, r_vocab)
  label_len = len(label)
  acc, acc1 = 0, 0
  num = 0
  for i in range(label_len):
    if not lpr_pattern_check(label[i].decode('utf-8'), lpr_patterns):  # GT label fails
      print('GT label fails: ' + label[i].decode('utf-8'))
      continue
    best = pred[i]
    edd = edit_distance(encode(label[i].decode('utf-8'), vocab), encode(best, vocab))
    if edd <= 1:
      acc1 += 1
    if label[i].decode('utf-8') == best:
      acc += 1
    else:
      if label[i].decode('utf-8') not in pred[i]:
        print('Check GT label: ' + label[i].decode('utf-8'))
      print(label[i].decode('utf-8') + ' -- ' + best + ' Edit Distance: ' + str(edd))
    num += 1
  return float(acc), float(acc1), num
