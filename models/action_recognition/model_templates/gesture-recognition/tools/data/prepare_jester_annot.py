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

from os import listdir
from os.path import exists, join, isfile
from argparse import ArgumentParser

from tqdm import tqdm


def load_label_map(file_path):
    action_names = []
    with open(file_path) as input_stream:
        for line in input_stream:
            line_parts = line.strip().split(';')
            if len(line_parts) != 1:
                continue

            action_name = line_parts[0]
            action_names.append(action_name)

    assert len(action_names) > 0

    unique_names = set(action_names)
    assert len(unique_names) == len(action_names)

    out_data = {name: label for label, name in enumerate(action_names)}

    return out_data


def load_annotation(annot_path, label_map):
    out_data = []
    with open(annot_path) as input_stream:
        for line in input_stream:
            line_parts = line.strip().split(';')
            if len(line_parts) != 2:
                continue

            rel_path, action_name = line_parts
            assert action_name in label_map

            action_id = label_map[action_name]

            out_data.append((rel_path, action_id))

    return out_data


def convert_annotation(src_annot, images_root, continuous_format):
    out_data = []
    for rel_path, action_id in tqdm(src_annot):
        images_dir = join(images_root, rel_path)
        if not exists(images_dir):
            continue

        files = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]
        if len(files) <= 1:
            continue

        frame_ids = [int(f.split('.')[0]) for f in files]
        assert min(frame_ids) == 1
        assert len(frame_ids) == max(frame_ids)

        num_frames = len(frame_ids)
        if continuous_format:
            out_data.append((rel_path, action_id, 0, num_frames - 1, 0, num_frames - 1, 30.0))
        else:
            out_data.append((rel_path, num_frames, action_id))

    return out_data


def dump_annotation(annot, out_path):
    with open(out_path, 'w') as output_stream:
        for record in annot:
            output_stream.write(' '.join([str(r) for r in record]) + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--label_map', '-lm', type=str, required=True)
    parser.add_argument('--images_root', '-im', type=str, required=True)
    parser.add_argument('--input_annot', '-ia', type=str, required=True)
    parser.add_argument('--out_annot', '-oa', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.label_map)
    assert exists(args.images_root)
    assert exists(args.input_annot)

    label_map = load_label_map(args.label_map)
    print('Loaded names for {} labels'.format(len(label_map)))

    annot = load_annotation(args.input_annot, label_map)
    print('Found {} records'.format(len(annot)))

    converted_annot = convert_annotation(annot, args.images_root, continuous_format=True)
    print('Converted {} / {} records'.format(len(converted_annot), len(annot)))

    dump_annotation(converted_annot, args.out_annot)
    print('Converted annotation is stored at: {}'.format(args.out_annot))


if __name__ == '__main__':
    main()
