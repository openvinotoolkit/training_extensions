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
import fnmatch
import logging
import os
import shlex
import sys
import tempfile
import unittest
import yaml

from collections import Counter
from copy import copy
from subprocess import run
from texttable import Texttable


KNOWN_DOMAIN_FOLDERS = ['object_detection', 'action_recognition_2', 'ote']
TEST_FILES_PATTERN = '*_tests_*.py'
MODEL_TEMPLATES_FOLDER_NAME = 'model_templates'
MODEL_TEMPLATES_FILE_NAME = 'template.yaml'
VENV_FOLDER_NAME = 'venv'


def run_with_log(cmd, check):
    cmdstr = ' '.join(cmd) if isinstance(cmd, list) else cmd
    cmdstr = (cmdstr.replace(';', ';\n')
              .replace(' -', ' \\\n    -')
              .replace(' "-', ' \\\n    "-'))
    logging.info(f'Running command\n`{cmdstr}`')
    return run(cmd, shell=True, check=check, executable="/bin/bash")

def _collect_all_tests(test_el):
    if isinstance(test_el, unittest.TestCase):
        return [test_el]
    all_tests = []
    for tst in test_el:
        all_tests.extend(_collect_all_tests(tst))
    return all_tests

def discover_all_tests(root_path):
    all_tests = []
    for cur_domain in KNOWN_DOMAIN_FOLDERS:
        cur_test_folder = os.path.join(root_path, cur_domain, 'tests')
        logging.debug(f'discover_all_tests: cur_test_folder={cur_test_folder}, TEST_FILES_PATTERN={TEST_FILES_PATTERN}')
        testsuite = unittest.TestLoader().discover(cur_test_folder, pattern=TEST_FILES_PATTERN)

        cur_tests = _collect_all_tests(testsuite)

        for tst in cur_tests:
            domain = getattr(tst, 'domain', cur_domain)
            el = {
                'test': tst,
                'domain': domain,
                'problem': getattr(tst, 'problem', None),
                'model': getattr(tst, 'model', None),
                'id': tst.id(),
                'topic': getattr(tst, 'topic', None),
            }
            all_tests.append(el)

    return all_tests

def _format_id_str(val):
    return '.\n'.join(val.split('.'))

def _add_counter_to_rows(rows):
    rows = [tuple(r) for r in rows]
    c = Counter(rows)
    new_rows = []
    for r in rows:
        if r not in c:
            continue
        new_rows.append(list(r) + [c[r],])
        del c[r]

    return new_rows

def print_list_tests(all_tests, verbose=False):
    keys = ['domain', 'problem', 'model', 'topic']

    rows = []
    info = {}
    for el in all_tests:
        row = [el[k] for k in keys]
        if verbose:
            row += [_format_id_str(el['id'])]
        rows.append(row)
        (info.setdefault(el['domain'], {})
                .setdefault(el['problem'], {})
                .setdefault(el['topic'], [])
                .append(el['model']))
    info_rows = [['domain', 'problem', 'topic', 'Num models', 'Num tests']]
    order_topics = ['train', 'export', 'nncf']
    for domain, info_domain in info.items():
        for problem, info_problem in info_domain.items():
            cur_topics = list(info_problem.keys())
            sorted_topics = [v for v in order_topics if v in cur_topics]
            sorted_topics += [v for v in cur_topics if v not in order_topics]
            for topic in sorted_topics:
                info_topic = info_problem[topic]
                num_tests = len(info_topic)
                num_models = len(set(info_topic))
                info_rows.append([domain, problem, topic, num_models, num_tests])

    table_info = Texttable(max_width=140)
    table_info.set_cols_align(['l', 'l', 'l', 'r', 'r'])
    table_info.set_deco(Texttable.HEADER)
    table_info.add_rows(info_rows)
    print('Info')
    print(table_info.draw())
    print()
    print('Details')
    if not verbose:
        print('(add flag --verbose for full list)')
    table = Texttable(max_width=140)
    table.set_deco(Texttable.HEADER)
    if not verbose:
        keys += ['Num tests']
        rows = _add_counter_to_rows(rows)
        table.set_cols_align(['l', 'l', 'l', 'l', 'r'])
        table.set_deco(Texttable.HEADER)
    else:
        keys += ['id']
        table.set_deco(Texttable.HEADER | Texttable.HLINES)
    rows = [keys] + rows
    table.add_rows(rows)
    print(table.draw())

def make_pattern_match(pattern, val):
    # TODO(LeonidBeynenson): add possibility to use regexp
    if val is None:
        return False
    res = fnmatch.fnmatchcase(val, pattern)
    return bool(res)

def filter_tests_by_pattern(all_tests, field_name, pattern):
    all_tests = [el for el in all_tests if make_pattern_match(pattern, el.get(field_name))]
    return all_tests

def filter_tests_by_value(all_tests, field_name, val):
    all_tests = [el for el in all_tests if el.get(field_name) == val]
    return all_tests

def get_domains_from_tests_list(all_tests):
    return sorted(set(el['domain'] for el in all_tests if el['domain']))

def write_list_template_files(root_path, all_tests, templates_list_file_path):
    template_files = []
    for el in all_tests:
        if None in (el['domain'], el['problem'], el['model']):
            continue
        template_path = os.path.join(root_path,
                                     el['domain'],
                                     MODEL_TEMPLATES_FOLDER_NAME,
                                     el['problem'],
                                     el['model'],
                                     MODEL_TEMPLATES_FILE_NAME)
        if not os.path.isfile(template_path):
            logger.warning(f'ATTENTION: cannot find template path {template_path}')
            continue
        template_files.append(template_path)
    with open(templates_list_file_path, 'w') as f:
        yaml.dump(template_files, f)

def instantiate_work_dir(root_path, work_dir, all_tests):
    os.makedirs(work_dir, exist_ok=True)

    _, tmp_f_name = tempfile.mkstemp(prefix='template_list_', suffix='.yaml', dir=work_dir)
    logging.info(f'Writing template files to temporary file {tmp_f_name}')

    write_list_template_files(root_path, all_tests, tmp_f_name)

    domains = get_domains_from_tests_list(all_tests)
    domains_str = ','.join(domains)

    run_with_log(f'cd {root_path}; python3 ./tools/instantiate.py'
                 f' --do-not-load-snapshots'
                 f' --templates-list-file {tmp_f_name}'
                 f' --domains {domains_str}'
                 f' {work_dir}',
                 check=True)

def is_in_virtual_env_in_work_dir(work_dir, domain):
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if not virtual_env:
        return False
    virtual_env = os.path.realpath(virtual_env)
    virtual_env = os.path.join(virtual_env, '') # for the last '/'
    work_domain = os.path.realpath(os.path.join(work_dir, domain))
    work_domain = os.path.join(work_domain, '')
    common = os.path.commonpath([work_domain, virtual_env])
    common = os.path.join(common, '')

    return common == work_domain

def generate_venv_path(work_dir, domain):
    return os.path.join(work_dir, domain, VENV_FOLDER_NAME)

def check_venvs(work_dir, all_tests):
    domains = get_domains_from_tests_list(all_tests)
    for domain in domains:
        venv_for_domain = generate_venv_path(work_dir, domain)
        assert os.path.isdir(venv_for_domain), \
                f'The venv folder {venv_for_domain} for domain {domain} is absent'

def run_testsuite(ts, work_dir, verbose):
    os.environ['MODEL_TEMPLATES'] = work_dir
    verbosity = 2 if verbose else 1
    r = unittest.TextTestRunner(verbosity=verbosity).run(ts)
    return r

def run_one_domain_tests_already_in_virtualenv(work_dir, all_tests, verbose):
    domains = get_domains_from_tests_list(all_tests)
    if len(domains) != 1:
        raise RuntimeError('The option --run-one-domain-inside-virtual-env may be used for one domain only')
    if not is_in_virtual_env_in_work_dir(work_dir, domains[0]):
        raise RuntimeError('The option --run-one-domain-inside-virtual-env may be used only'
                           ' inside the virtual environament of the domain')

    testsuite = unittest.TestSuite()
    for el in all_tests:
        testsuite.addTest(el['test'])

    tests_res = run_testsuite(testsuite, work_dir, verbose)

    was_successful = tests_res.wasSuccessful()
    sys_retval = int(not was_successful)
    sys.exit(sys_retval)

def rerun_inside_virtual_envs(work_dir, all_tests, args):
    def _success_to_str(was_successful):
        return 'OK' if was_successful else 'FAIL'

    domains = get_domains_from_tests_list(all_tests)
    results = {}
    for domain in domains:
        logging.info(f'Begin running tests for domain {domain}')
        new_argv = ['python3'] + sys.argv
        if not args.domain:
            new_argv.extend(['--domain', domain])
        new_argv.append('--run-one-domain-inside-virtual-env')

        assert all('"' not in v for v in new_argv), \
                f'Cannot work if arguments contain double quotes:\n{new_argv}'

        cmd = ' '.join(shlex.quote(v) for v in new_argv)
        venv_path = generate_venv_path(work_dir, domain)
        cmd = f'source "{venv_path}/bin/activate"; ' + cmd

        res = run_with_log(cmd, check=False)

        was_successful = (res.returncode == 0)
        results[domain] = was_successful
        logging.info(f'End running tests for domain {domain}, result={_success_to_str(was_successful)}')
        logging.info('')

    logging.info('~' * 70)
    logging.info('Result:')
    total_success = True
    for domain in domains:
        was_successful = results[domain]
        total_success = total_success and was_successful
        logging.info(f'    {domain}: {_success_to_str(was_successful)}')
    logging.info(f'Total: {_success_to_str(total_success)}')

def _get_pytorch_toolkit_path():
    cur_file_path = os.path.abspath(__file__)
    pytorch_toolkit_path = os.path.dirname(os.path.dirname(cur_file_path))
    return pytorch_toolkit_path

def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir',
                        help='The work folder where to instantiate tests and run them. '
                        'Optional. If absent, a temporary folder will be used')
    parser.add_argument('--domain', choices=KNOWN_DOMAIN_FOLDERS, help='Domain name to be tested.  Optional.')
    parser.add_argument('--problem', help='Problem name to be tested. Optional.')
    parser.add_argument('--problem-filter', help='Filter on problem name to be tested. Optional.')
    parser.add_argument('--topic', choices=['train', 'export', 'nncf', 'internal'], help='Topic of tests to be tested. Optional')
    parser.add_argument('--test-id-filter', action='append',
                        help='Filter on test id-s. Optional. Several filters are applied using logical AND')
    parser.add_argument('--verbose', '-v', action='store_true', help='If the tests should be run in verbose mode')
    parser.add_argument('--list', '-l', action='store_true', help='List all available tests')
    parser.add_argument('--instantiate-only', action='store_true', help='If the script should instantiate the tests in the work dir only')
    parser.add_argument('--not-instantiate', action='store_true', help='If the script should NOT instantiate the tests in the work dir')
    parser.add_argument('--run-one-domain-inside-virtual-env', action='store_true',
                        help='If the script should run the tests for one domain without work dir instantiation.'
                        ' It is supposed that the script is already run in the proper virtual environment.')


    args = parser.parse_args()
    assert not (args.instantiate_only and args.not_instantiate), \
            'Only one of parameters --instantiate-only and --not-instantiate may be set'
    assert not (args.instantiate_only and args.run_one_domain_inside_virtual_env), \
            'Only one of parameters --instantiate-only and --run-one-domain-only may be set'

    root_path = _get_pytorch_toolkit_path()
    all_tests = discover_all_tests(root_path)

    if args.domain:
        all_tests = filter_tests_by_value(all_tests, 'domain', args.domain)
    if args.topic:
        all_tests = filter_tests_by_value(all_tests, 'topic', args.topic)
    if args.problem:
        all_tests = filter_tests_by_value(all_tests, 'problem', args.problem)
    if args.problem_filter:
        all_tests = filter_tests_by_pattern(all_tests, 'problem', args.problem_filter)
    if args.test_id_filter:
        for test_filter in args.test_id_filter:
            all_tests = filter_tests_by_pattern(all_tests, 'id', test_filter)

    if args.list:
        print_list_tests(all_tests, args.verbose)
        return

    work_dir = os.path.abspath(args.workdir) if args.workdir else tempfile.mkstemp(prefix='work_dir_')
    logging.info(f'work_dir = {work_dir}')

    should_instantiate = (not args.run_one_domain_inside_virtual_env) and (not args.not_instantiate)
    if should_instantiate:
        instantiate_work_dir(root_path, work_dir, all_tests)
        logging.info(f'The work_dir {work_dir} is instantiated')
        check_venvs(work_dir, all_tests)
        logging.info('Instantiation checks are passed')

    if args.instantiate_only:
        return

    domains = get_domains_from_tests_list(all_tests)
    if args.run_one_domain_inside_virtual_env:
        run_one_domain_tests_already_in_virtualenv(work_dir, all_tests, args.verbose)
        return

    rerun_inside_virtual_envs(work_dir, all_tests, args)


if __name__ == '__main__':
    main()
