# Copyright (C) 2020 Intel Corporation
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

import argparse
import glob
import os
from subprocess import run

import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    parser.add_argument('--do-not-load-snapshots', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    template_filenames = glob.glob('**/template.yaml', recursive=True)
    problems_filename = glob.glob('**/problems.yaml', recursive=True)
    assert len(problems_filename) == 1
    problems_filename = problems_filename[0]

    problems_dict = dict()
    with open(problems_filename) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
        for domain in content['domains']:
            for problem in domain['problems']:
                problems_dict[problem['title']] = problem

    os.makedirs(args.destination, exist_ok=True)

    for template_filename in tqdm(template_filenames):
        with open(template_filename) as read_file:
            content = yaml.load(read_file, yaml.SafeLoader)

        problem_folder = os.path.join(
            args.destination, content['domain'], content['problem']).replace(' ', '_').lower()
        instance_folder = os.path.join(
            problem_folder, content['name']).replace(' ', '_').lower()

        if args.do_not_load_snapshots:
            run(f'python tools/instantiate_template.py {template_filename} {instance_folder}'
                f' --do-not-load-snapshot', check=True, shell=True)
        else:
            run(f'python tools/instantiate_template.py {template_filename} {instance_folder}',
                check=True, shell=True)

        problem_dict = problems_dict[content['problem']]
        with open(os.path.join(problem_folder, 'problem.yaml'), 'w') as write_file:
            yaml.dump(problem_dict, write_file)
        if problem_dict.get('type', None) != 'generic':
            with open(os.path.join(problem_folder, 'schema.json'), 'w') as write_file:
                write_file.write(problem_dict['cvat_schema'])


if __name__ == '__main__':
    main()
