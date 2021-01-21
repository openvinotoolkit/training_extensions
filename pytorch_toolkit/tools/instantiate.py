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
import logging
import glob
import logging
import os
import yaml

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ote'))

from ote.utils.misc import run_through_shell

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    parser.add_argument('--do-not-load-snapshots', action='store_true')
    parser.add_argument('--templates-filter', default='**/template.yaml')
    parser.add_argument('--templates-list-file',
                        help='A yaml file with list of paths of template files'
                         ' to be instantiated. Overrides --template-filter.')
    parser.add_argument('--domains',
                        help='Comma-separated list of domains that should be additionally'
                        ' instantiated even if there are no found templates inside them'
                        ' (should not contain spaces)')
    parser.add_argument('--verbose', '-v', action='store_true', help='If the instantiation should be run in verbose mode')

    return parser.parse_args()

def _get_templates_filenames(args):
    if args.templates_list_file:
        with open(args.templates_list_file) as f_t_list:
            template_filenames = yaml.safe_load(f_t_list)
    else:
        template_filenames = glob.glob(args.templates_filter, recursive=True)
        template_filenames = list(template_filenames)
    return template_filenames

def _find_init_venv(domain_folder):
    domain_realpath = os.path.realpath(domain_folder)

    # the second candidate is used if virtual environment is used for internal tests only
    candidate_init_venv_1 = os.path.join(domain_realpath, 'init_venv.sh')
    candidate_init_venv_2 = os.path.join(domain_realpath, 'tests', 'init_venv.sh')

    if os.path.isfile(candidate_init_venv_1):
        return candidate_init_venv_1
    if os.path.isfile(candidate_init_venv_2):
        return candidate_init_venv_2

    raise RuntimeError(f'Cannot find init_venv.sh script in domain {domain_folder} in'
                       f' {candidate_init_venv_1} or {candidate_init_venv_2}')

def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    template_filenames = _get_templates_filenames(args)
    if args.domains:
        additional_domains = args.domains.split(',')
    else:
        additional_domains = []

    problems_filename = glob.glob('**/problems.yaml', recursive=True)
    assert len(problems_filename) == 1
    problems_filename = problems_filename[0]

    problems_dict = dict()
    with open(problems_filename) as read_file:
        content = yaml.safe_load(read_file)
        for domain in content['domains']:
            for problem in domain['problems']:
                problems_dict[problem['title']] = problem

    os.makedirs(args.destination, exist_ok=True)

    domain_folders = set()
    for template_filename in template_filenames:
        with open(template_filename) as read_file:
            content = yaml.safe_load(read_file)

        # TODO(ikrylov): remain one of ('-', '_').
        domain_folder = content['domain'].replace(' ', '_').lower()
        problem_folder = content['problem'].replace(' ', '-').lower()
        model_folder = content['name']

        domain_folders.add(domain_folder)

        problem_folder = os.path.join(args.destination, domain_folder, problem_folder)
        instance_folder = os.path.join(problem_folder, model_folder)

        logging.debug(f'Begin instantiating {template_filename} to {instance_folder}')
        do_not_load_snapshot_str = ' --do-not-load-snapshot' if args.do_not_load_snapshots else ''
        run_through_shell(f'python3 tools/instantiate_template.py {template_filename} {instance_folder}'
                          f'{do_not_load_snapshot_str}',
                          verbose=args.verbose)
        logging.debug(f'End instantiating {template_filename} to {instance_folder}')

        problem_dict = problems_dict.get(content['problem'], None)
        if problem_dict is None:
            logging.warning(f'The {content["problem"]} is not listed in {problems_filename}. It will not be in Web UI.')
        else:
            with open(os.path.join(problem_folder, 'problem.yaml'), 'w') as write_file:
                yaml.dump(problem_dict, write_file)
            if problem_dict.get('type', None) != 'generic':
                with open(os.path.join(problem_folder, 'schema.json'), 'w') as write_file:
                    write_file.write(problem_dict['cvat_schema'])

    logging.info(f'Instantiated {len(template_filenames)} templates')

    domain_folders = domain_folders | set(additional_domains)
    for domain_folder in domain_folders:
        logging.info(f'Begin initializing virtual environment for {domain_folder}')
        init_venv_path = _find_init_venv(domain_folder)
        run_through_shell(f'cd {domain_folder}; '
                          f'bash {init_venv_path} {os.path.join(args.destination, domain_folder, "venv")}',
                          verbose=args.verbose)
        logging.info(f'End initializing virtual environment for {domain_folder}')

if __name__ == '__main__':
    main()
