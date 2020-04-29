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

import sys
from argparse import ArgumentParser

import torch
from os import listdir, makedirs
from os.path import isfile, join, exists


def main(argv):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-folder', help='Path to directory with given checkpoints to modify',
                        required=True)
    parser.add_argument('-r', '--ref-folder', help='Path to directory with reference (new) checkpoints',
                        required=True)
    parser.add_argument('-o', '--output-folder', help='Path to directory to save modified checkpoints', required=True)
    args = parser.parse_args(args=argv)

    src_dir = args.input_folder
    dst_dir = args.output_folder
    ref_dir = args.ref_folder
    if not exists(dst_dir):
        makedirs(dst_dir)

    pth_files = [(join(src_dir, f), join(dst_dir, f), join(ref_dir, f)) for f in listdir(src_dir) if
                 isfile(join(src_dir, f)) and ('.pth' in f or '.sd' in f)]

    for src_file, dst_file, ref_file in pth_files:
        ref_sd = torch.load(ref_file)
        if 'state_dict' in ref_sd:
            ref_sd = ref_sd['state_dict']


        sd = pth = torch.load(src_file)
        if 'state_dict' in pth:
            sd = pth['state_dict']

        ref_keys = list(sorted(list(ref_sd.keys())))
        old_keys = list(sorted(list(sd.keys())))

        old_to_ref_map = {}

        old_mean_0_keys = list(filter(lambda x: "mean_0" in x, old_keys))
        old_mean_1_keys = list(filter(lambda x: "mean_1" in x, old_keys))
        new_mean_0_keys = list(filter(lambda x: "mean_0" in x, ref_keys))

        old_keys = list(filter(lambda x: x not in old_mean_0_keys, old_keys))
        for idx, old_mean_1_key in enumerate(old_mean_1_keys):
            old_to_ref_map[old_mean_1_key] = new_mean_0_keys[idx]
            old_keys.remove(old_mean_1_key)
            ref_keys.remove(new_mean_0_keys[idx])

        first_weight_related_old_key = next(x for x in old_keys if "activation_quantizers" not in x)
        first_weight_related_old_idx = old_keys.index(first_weight_related_old_key)

        old_act_keys = old_keys[:first_weight_related_old_idx]
        old_weight_keys = old_keys[first_weight_related_old_idx:]

        ref_act_keys = ref_keys[:first_weight_related_old_idx]
        ref_weight_keys = ref_keys[first_weight_related_old_idx:]

        old_base_act_keys = list(filter(lambda x: "Inverted" in x, old_act_keys))
        ref_base_act_keys = list(filter(lambda x: "Inverted" in x, ref_act_keys))

        old_other_act_keys = list(filter(lambda x: x not in old_base_act_keys, old_act_keys))
        ref_other_act_keys = list(filter(lambda x: x not in ref_base_act_keys, ref_act_keys))

        for idx, old_key in enumerate(old_weight_keys):
            old_to_ref_map[old_key] = ref_weight_keys[idx]

        for idx, old_key in enumerate(old_base_act_keys):
            old_to_ref_map[old_key] = ref_base_act_keys[idx]

        for idx, old_key in enumerate(old_other_act_keys):
            old_to_ref_map[old_key] = ref_other_act_keys[idx]

        assert len(old_to_ref_map) == len(ref_sd)

        new_sd = {}

        for old_key, ref_key in old_to_ref_map.items():
            new_sd[ref_key] = sd[old_key]

        pth['state_dict'] = new_sd
        torch.save(pth, dst_file)

if __name__ == '__main__':
    main(sys.argv[1:])
