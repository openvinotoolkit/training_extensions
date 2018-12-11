from __future__ import print_function
import re
from lpr.trainer import encode, decode_beams

def dataset_size(fname):
  count = 0
  with open(fname, 'r') as file_:
    for _ in file_:
      count += 1
  return count


LPR_PATTERNS = [
  '^<[^>]*>[A-Z][0-9A-Z]{5}$',
  '^<[^>]*>[A-Z][0-9A-Z][0-9]{3}<police>$',
  '^<[^>]*>[A-Z][0-9A-Z]{4}<[^>]*>$',  # <Guangdong>, <Hebei>
  '^WJ<[^>]*>[0-9]{4}[0-9A-Z]$',
]

def lpr_pattern_check(label):
  for pattern in LPR_PATTERNS:
    if re.match(pattern, label):
      return True
  return False

def find_best(predictions):
  for prediction in predictions:
    if lpr_pattern_check(prediction):
      return prediction
  return predictions[0]  # fallback


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


def accuracy(label, val, vocab, r_vocab):
  pred = decode_beams(val, r_vocab)
  label_len = len(label)
  acc, acc1 = 0, 0
  num = 0
  for i in range(label_len):
    if not lpr_pattern_check(label[i].decode('utf-8')):  # GT label fails
      print('GT label fails: ' + label[i].decode('utf-8'))
      continue
    best = find_best(pred[i])
    # use classes lists instead of strings to get edd('<aaa>', '<bbb>') = 1
    edd = edit_distance(encode(label[i].decode('utf-8'), vocab), encode(best, vocab))
    if edd <= 1:
      acc1 += 1
    if label[i].decode('utf-8') == best:
      acc += 1
    else:
      if label[i] not in pred[i]:
        print('Check GT label: ' + label[i].decode('utf-8'))
      print(label[i].decode('utf-8') + ' -- ' + best + ' Edit Distance: ' + str(edd))
    num += 1
  return float(acc), float(acc1), num
