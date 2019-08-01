"""
 Copyright (c) 2019 Intel Corporation
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

import argparse
import os
from os.path import join, isdir, isfile

import torch

from tools.debug.common import print_args

argparser = argparse.ArgumentParser()
argparser.add_argument("-r", "--ref-dir",
                       help="Path to ref folder. Treated whether ref experiment, or dir to compare GPU dumps",
                       required=True)
argparser.add_argument("-c", "--cmp-dirs",
                       help="List of path to dirs with experiments for comparison with ref folder 0,1,2...", nargs='+')
argparser.add_argument("--range", type=int, help="Right border of range, starting from ref dir number", )
argparser.add_argument("--eps", help="Torelance for maximum of absolute difference", default=None, type=float)

args = argparser.parse_args()
print_args(args)


def basename(x):
    return os.path.basename(x)


def get_dirs(root_path):
    return [join(root_path, f) for f in sorted(os.listdir(root_path), key=basename) if isdir(join(root_path, f))]


def get_files(root_path):
    return [f for f in sorted(os.listdir(root_path), key=basename) if isfile(join(root_path, f))]


def get_dir_pairs(dump_dir):
    dirs = get_dirs(dump_dir)
    return [(dirs[0], cmp_dir) for cmp_dir in dirs[1:]]


def compare_dump_in_dir_pairs(pairs, eps=None):
    global_max_diff = 0
    global_count_diff = 0
    for ref_dir, cmp_dir in pairs:
        max_diff = compare_dump_in_file_pairs(ref_dir, cmp_dir, eps)
        if max_diff != 0:
            global_count_diff += 1
        if max_diff > global_max_diff:
            global_max_diff = max_diff
    print('\n\nGlobal MAX abs diff: {}\n{}/{} is different'.format(global_max_diff, global_count_diff, len(pairs)))


def compare_dump_in_file_pairs(ref_dir, cmp_dir, eps):
    ref_files = get_files(ref_dir)
    cmp_files = get_files(cmp_dir)
    max_diff = 0
    count_diff = 0
    print('\n\nCompare {} vs {}'.format(ref_dir, cmp_dir))
    for rf in ref_files:
        rt = torch.load(os.path.join(ref_dir, rf))
        if rf in cmp_files:
            ct = torch.load(os.path.join(cmp_dir, rf))
            rn = os.path.basename(rf)
            diff = abs(rt - ct).max()
            if diff != 0:
                count_diff += 1
            if diff > max_diff:
                max_diff = diff
            if eps is not None:
                if diff >= eps:
                    if 'scale' in rf:
                        print('____{} vs {}_____, diff={} for {}'.format(rt.item(), ct.item(), diff, rn))
                    else:
                        print('diff={} for {}'.format(diff, rn))

        else:
            print('not matched file {}'.format(rf))
    print('Max abs diff: {}\n{}/{} is different'.format(max_diff, count_diff, len(ref_files)))
    return max_diff


def main():
    ref_dir = args.ref_dir
    cmp_dirs = args.cmp_dirs
    range_ = args.range
    dir_pairs = []
    if cmp_dirs:
        for cmp_dir in cmp_dirs:
            dir_pairs += list(zip(get_dirs(cmp_dir), get_dirs(ref_dir)))
    elif range_:
        lb = int(os.path.basename(ref_dir))
        parent_dir = os.path.abspath(os.path.join(ref_dir, os.pardir))
        for i in range(lb + 1, range_ + 1):
            cmp_dir = os.path.join(parent_dir, str(i))
            dir_pairs += list(zip(get_dirs(cmp_dir), get_dirs(ref_dir)))
    else:
        dir_pairs = get_dir_pairs(ref_dir)

    compare_dump_in_dir_pairs(dir_pairs, args.eps)


if __name__ == '__main__':
    main()
