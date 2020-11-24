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
import logging
import os
from subprocess import run

import yaml

def run_with_log(*args, **kwargs):
    cmd = args[0]
    logging.info(f'Running command\n`{cmd}`') #TODO: consider with Ilya
    return run(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    parser.add_argument('--do-not-load-snapshots', action='store_true')
    parser.add_argument('--templates-pattern', default='**/template.yaml')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    template_filenames = glob.glob(args.templates_pattern, recursive=True)
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

    domain_folders = set()
    for template_filename in template_filenames:
        with open(template_filename) as read_file:
            content = yaml.load(read_file, yaml.SafeLoader)

        # TODO(ikrylov): remain one of ('-', '_').
        domain_folder = content['domain'].replace(' ', '_').lower()
        problem_folder = content['problem'].replace(' ', '-').lower()
        model_folder = content['name']

        domain_folders.add(domain_folder)

        problem_folder = os.path.join(args.destination, domain_folder, problem_folder)
        instance_folder = os.path.join(problem_folder, model_folder)

        logging.info(f'Instantiate {template_filename} to {instance_folder}')
        if args.do_not_load_snapshots:
            run_with_log(f'python3 tools/instantiate_template.py {template_filename} {instance_folder}'
                f' --do-not-load-snapshot', check=True, shell=True)
        else:
            run_with_log(f'python3 tools/instantiate_template.py {template_filename} {instance_folder}',
                check=True, shell=True)

        problem_dict = problems_dict[content['problem']]
        with open(os.path.join(problem_folder, 'problem.yaml'), 'w') as write_file:
            yaml.dump(problem_dict, write_file)
        if problem_dict.get('type', None) != 'generic':
            with open(os.path.join(problem_folder, 'schema.json'), 'w') as write_file:
                write_file.write(problem_dict['cvat_schema'])

    for domain_folder in domain_folders:
        run_with_log(f'cd {domain_folder}; ./init_venv.sh {os.path.join(args.destination, domain_folder, "venv")}', shell=True, check=True)

if __name__ == '__main__':
    main()
