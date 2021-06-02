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

import yaml
from ote_cli.utils.misc import run_through_shell


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='model_templates')
    parser.add_argument('--templates-filter', default='external/**/template.yaml')
    parser.add_argument('--templates-list-file',
                        help='A yaml file with list of paths of template files'
                         ' to be instantiated. Overrides --template-filter.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If the instantiation should be run in verbose mode')
    parser.add_argument('--init-venv', action='store_true')

    return parser.parse_args()

def _get_templates_filenames(args):
    if args.templates_list_file:
        with open(args.templates_list_file) as f_t_list:
            template_filenames = yaml.safe_load(f_t_list)
    else:
        template_filenames = glob.glob(args.templates_filter, recursive=True)
        template_filenames = list(template_filenames)
    return template_filenames

def _get_init_venv_path(algo_repo_name):
    algo_repo_name = os.path.realpath(os.path.join('external', algo_repo_name))
    init_venv_path = os.path.join(algo_repo_name, 'init_venv.sh')
    if os.path.isfile(init_venv_path):
        return init_venv_path
    return None

def main():
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    template_filenames = _get_templates_filenames(args)

    os.makedirs(args.destination, exist_ok=True)

    repo_name_to_framework_folder = dict()
    for template_filename in template_filenames:
        with open(template_filename) as read_file:
            content = yaml.safe_load(read_file)

        assert template_filename.startswith('external'), f'{template_filename}'
        algo_repo_name = template_filename.split(os.sep)[1]

        framework_folder = content['framework'].replace(' ', '_')
        domain_folder = content['domain'].replace(' ', '_')
        model_folder = content['name'].replace(' ', '_')

        repo_name_to_framework_folder[algo_repo_name] = framework_folder

        instance_folder = os.path.join(args.destination, framework_folder, domain_folder, model_folder)

        logging.debug(f'Begin instantiating {template_filename} to {instance_folder}')
        run_through_shell(f'python3 tools/instantiate_template.py {template_filename} {instance_folder}',
                          verbose=args.verbose)
        logging.debug(f'End instantiating {template_filename} to {instance_folder}')

    logging.info(f'Instantiated {len(template_filenames)} templates')

    if args.init_venv:
        for algo_repo_name, framework_folder in repo_name_to_framework_folder.items():
            logging.info(f'Begin initializing virtual environment for {algo_repo_name}:{framework_folder}')
            dst_domain_path = os.path.join(args.destination, framework_folder)
            print(algo_repo_name)
            init_venv_path = _get_init_venv_path(algo_repo_name)
            if init_venv_path is None:
                logging.info(f'    No virtual environment for {algo_repo_name}')
                continue
            dst_venv_path = os.path.abspath(os.path.join(dst_domain_path, 'venv'))
            run_through_shell(f'bash {init_venv_path} {dst_venv_path}',
                            verbose=args.verbose)

            venv_activate_path = os.path.join(dst_venv_path, 'bin', 'activate')
            run_through_shell(f'source {venv_activate_path};'
                              'pip install -e ote_cli/;'
                              'pip install notebook;'
                              f'cd $SC_SDK_REPO/src; python sc_sdk/setup.py develop',
                              verbose=args.verbose)
            logging.info(f'End initializing virtual environment for {algo_repo_name}.\n'
                         f'To activate run: \'source {venv_activate_path}\'')

if __name__ == '__main__':
    main()
