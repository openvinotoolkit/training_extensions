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
from os.path import exists, join, isfile, isdir
from argparse import ArgumentParser

from tqdm import tqdm


VALID_EXTENSIONS = ['jpg', 'jpeg', 'png']


def parse_image_paths(root_dir):
    image_paths = []
    for class_dir in tqdm(listdir(root_dir)):
        class_dir_path = join(root_dir, class_dir)
        if isdir(class_dir_path):
            for file_name in listdir(class_dir_path):
                file_path = join(class_dir_path, file_name)
                file_extension = file_name.split('.')[-1].lower()
                if isfile(file_path) and file_extension in VALID_EXTENSIONS:
                    image_paths.append(join(class_dir, file_name))

    return image_paths


def dump_image_paths(paths, out_file_path):
    with open(out_file_path, 'w') as output_stream:
        for file_path in paths:
            output_stream.write('{}\n'.format(file_path))


def main():
    parser = ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    assert exists(args.root_dir)

    image_paths = parse_image_paths(args.root_dir)
    print('Found {} images'.format(len(image_paths)))

    dump_image_paths(image_paths, args.output_file)
    print('Stored at: {}'.format(args.output_file))


if __name__ == '__main__':
    main()
