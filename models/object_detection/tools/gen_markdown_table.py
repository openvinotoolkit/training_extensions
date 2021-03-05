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

""" This script builds markdown table with information about models. """

import argparse
from collections import OrderedDict
import glob
import yaml


def parse_args():
    """ Parses input args. """

    args = argparse.ArgumentParser()
    args.add_argument('problem_folder')
    return args.parse_args()


def get_header(metrics):
    """ Get header name from metrics dict. """

    return f'{metrics["display_name"]} ({metrics["unit"]})'


def extract_info(descr):
    """ Extracts model infor from model.yml content. """

    info = OrderedDict()
    info.update({'Model Name': descr['name']})

    complexity = [metrics for metrics in descr['metrics'] if metrics['key'] == 'complexity'][0]
    info.update({get_header(complexity): complexity['value']})

    size = [metrics for metrics in descr['metrics'] if metrics['key'] == 'size'][0]
    info.update({get_header(size): size['value']})

    for metrics in descr['metrics']:
        if metrics['key'] not in ['size', 'complexity']:
            info.update({get_header(metrics): metrics['value']})

    info.update(
        {'Links': f'[snapshot]({descr["files"][0]["source"]}), [configuration file](./{descr["name"]}/config.py)'})

    info.update({'GPU_NUM': descr['training_gpu_num']})

    return info


def build_table(infos):
    """ Builds markdown table. """

    table_str = '| '
    for key in infos[0].keys():
        table_str += key + ' | '
    table_str += '\n'

    table_str += '| '
    for key in infos[0].keys():
        table_str += '--- | '
    table_str += '\n'

    for info in infos:
        table_str += '| '
        for value in info.values():
            table_str += str(value) + ' | '
        table_str += '\n'

    return table_str


def main():
    """ Main function. """

    args = parse_args()
    models_infos = []
    for x in sorted(glob.glob(f'{args.problem_folder}/**/model.yml', recursive=True)):
        with open(x) as read_file:
            models_infos.append(extract_info(yaml.load(read_file)))

    keys = models_infos[0].keys()

    for info in models_infos:
        assert keys == info.keys(), f'{keys} does not equal to {info.keys()}'

    print(build_table(models_infos))


if __name__ == '__main__':
    main()
