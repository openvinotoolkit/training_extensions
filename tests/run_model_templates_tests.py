# Copyright (C) 2020-2021 Intel Corporation
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
import glob
import inspect
import json
import logging
import os
import shlex
import sys
import tempfile
import unittest
import yaml

from collections import Counter
from copy import deepcopy
from subprocess import run
from texttable import Texttable


KNOWN_DOMAIN_FOLDERS = [
        'object_detection',
        'action_recognition',
        'instance_segmentation',
        'text_spotting',
        'image_classification',
        ]
TEST_FILES_PATTERN = '*_tests_*.py'
MODEL_TEMPLATES_FOLDER_NAME = 'model_templates'
MODEL_TEMPLATES_FILE_NAME = 'template.yaml'
VENV_FOLDER_NAME = 'venv'


def run_with_log(cmd, check):
    cmdstr = ' '.join(cmd) if isinstance(cmd, list) else cmd
    cmdstr = (cmdstr.replace(';', ';\n')
              .replace(' -', ' \\\n    -')
              .replace(' "-', ' \\\n    "-'))
    logging.debug(f'Running command\n`{cmdstr}`')
    return run(cmd, shell=True, check=check, executable="/bin/bash")

def _collect_all_tests(test_el):
    if isinstance(test_el, unittest.TestCase):
        return [test_el]
    all_tests = []
    for tst in test_el:
        all_tests.extend(_collect_all_tests(tst))
    return all_tests

def _get_domains(restrict_to_domain=None):
    if restrict_to_domain:
        assert restrict_to_domain in KNOWN_DOMAIN_FOLDERS, (
                f'Error: unknown domain "{restrict_to_domain}",'
                f' known domains are {KNOWN_DOMAIN_FOLDERS}')
        return [restrict_to_domain]

    return KNOWN_DOMAIN_FOLDERS

def _find_all_templates(root_path, restrict_to_domain=None):
    logging.debug('Begin running discovery of templates')
    all_domains = _get_domains(restrict_to_domain)
    all_templates = []
    for cur_domain in all_domains:
        domain_path = os.path.join(root_path, cur_domain)
        all_templates += list(glob.glob(f'{domain_path}/**/{MODEL_TEMPLATES_FILE_NAME}', recursive=True))
    logging.info(f'Loaded {len(all_templates)} templates from {len(all_domains)} domains')
    logging.debug('End running discovery of templates')
    return all_templates

def find_all_model_templates(root_path, restrict_to_domain=None):
    logging.debug('Begin building map model_name to folder')
    all_templates = _find_all_templates(root_path, restrict_to_domain)
    model_name_to_template = {}
    for tmpl_path in all_templates:
        with open(tmpl_path) as f:
            content = yaml.safe_load(f)
            name = content['name']
            if name in model_name_to_template:
                raise RuntimeError(f'Error: duplication of model name {name}:'
                                   f' {model_name_to_template[name]} and {tmpl_path}')
            model_name_to_template[name] = tmpl_path
    logging.info(f'Loaded {len(model_name_to_template)} models')
    logging.debug('End building map model_name to folder')
    return model_name_to_template

def discover_all_tests(root_path, restrict_to_domain=None):
    logging.debug('Begin running discovery of tests')
    all_domains = _get_domains(restrict_to_domain)

    all_tests = []
    failed_modules = []
    for cur_domain in all_domains:
        cur_test_folder = os.path.join(root_path, cur_domain, 'tests')
        testsuite = unittest.TestLoader().discover(cur_test_folder, pattern=TEST_FILES_PATTERN)

        cur_tests = _collect_all_tests(testsuite)

        for tst in cur_tests:
            domain = getattr(tst, 'domain', cur_domain)
            problem = getattr(tst, 'problem', None)
            el = {
                'test': tst,
                'domain': domain,
                'problem': problem,
                'model': getattr(tst, 'model', None),
                'id': tst.id(),
                'topic': getattr(tst, 'topic', None),
            }
            if isinstance(tst, unittest.loader._FailedTest): # pylint: disable=protected-access
                logging.warning(f'Failed to load test {el}:\n{tst._exception}') # pylint: disable=protected-access
                cur_id = tst.id()
                failed_prefix = 'unittest.loader._FailedTest.'
                if cur_id.startswith(failed_prefix):
                    cur_id = cur_id[len(failed_prefix):]
                failed_modules.append(cur_id)
            all_tests.append(el)

    logging.debug('End running discovery of tests')
    return all_tests, failed_modules

def fill_template_paths_in_test_elements(all_tests, model_name_to_template):
    for el in all_tests:
        model_name = el['model']
        if model_name is None:
            el['template_path'] = None
            continue
        if model_name not in model_name_to_template:
            raise RuntimeError(f'Error: model name {model_name} not present in templates')
        el['template_path'] = model_name_to_template[model_name]

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

def save_tests_list(list_of_tests, path):
    if not path:
        raise RuntimeError('Try to write the list of tests to an empty path')

    list_of_tests = deepcopy(list_of_tests)
    for el in list_of_tests:
        del el['test']

    ext = os.path.splitext(path)[1]
    with open(path, 'w') as f:
        if ext in ('.yaml', '.yml'):
            yaml.dump(list_of_tests, f)
            return
        if ext in ('.json', '.jsn'):
            json.dump(list_of_tests, f, indent=4)
            return
    raise RuntimeError(f'Unknown extension of path {path}')

def print_list_tests(all_tests, verbose=False, short=False):
    assert not (verbose and short)
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
    print(table_info.draw(), flush=True)
    if short:
        return
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
    print(table.draw(), flush=True)

def make_pattern_match(pattern, val):
    # TODO(LeonidBeynenson): add possibility to use regexp
    if val is None:
        return False
    res = fnmatch.fnmatchcase(val, pattern)
    return bool(res)

def filter_tests_by_pattern(all_tests, field_name, pattern):
    all_tests = [el for el in all_tests if make_pattern_match(pattern, el.get(field_name))]
    return all_tests

def filter_tests_by_many_patterns(all_tests, field_name, patterns):
    new_tests_ids = set()
    for pattern in patterns:
        filtered_tests = filter_tests_by_pattern(all_tests, field_name, pattern)
        filtered_tests_ids = {el['id'] for el in filtered_tests}
        new_tests_ids.update(filtered_tests_ids)
    all_tests = [el for el in all_tests if el['id'] in new_tests_ids]
    return all_tests

def filter_tests_by_value(all_tests, field_name, val):
    all_tests = [el for el in all_tests if el.get(field_name) == val]
    return all_tests

def get_domains_from_tests_list(all_tests):
    return sorted(set(el['domain'] for el in all_tests if el['domain']))

def write_list_template_files(root_path, all_tests, templates_list_file_path):
    template_files = []
    for el in all_tests:
        template_path = el['template_path']
        if template_path is None:
            continue
        if not os.path.isfile(template_path):
            raise RuntimeError(f'Cannot find template path {template_path}')
        template_files.append(template_path)

    # usually there are many tests for one template
    template_files = sorted(set(template_files))

    with open(templates_list_file_path, 'w') as f:
        yaml.dump(template_files, f)

def instantiate_work_dir(root_path, work_dir, all_tests, verbose):
    os.makedirs(work_dir, exist_ok=True)

    _, tmp_f_name = tempfile.mkstemp(prefix='template_list_', suffix='.yaml', dir=work_dir)
    logging.debug(f'Writing template files to temporary file {tmp_f_name}')

    write_list_template_files(root_path, all_tests, tmp_f_name)

    verbose_param = ' --verbose' if verbose else ''

    run_with_log(f'python3 ./tools/instantiate.py'
                 f' --do-not-load-snapshots'
                 f' --templates-list-file {tmp_f_name}'
                 f' {work_dir}'
                 f' {verbose_param}',
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

def pytest_run_tests(all_tests, work_dir, verbose, should_capture_output=True):
    pytest_ids = []
    for el in all_tests:
        curfile = inspect.getfile(type(el['test']))
        curfile = os.path.relpath(curfile)
        classname = type(el['test']).__name__
        cur_id = el['id']
        cur_id = cur_id[cur_id.rfind('.')+1:]
        cur_pytest_id = f'{curfile}::{classname}::{cur_id}'
        pytest_ids.append(cur_pytest_id)

    pytest_ids = ' '.join(pytest_ids)

    if verbose and not should_capture_output:
        verb_flag = '-v -o log_cli=true -o log_cli_level=DEBUG'
    elif verbose:
        verb_flag = '-v'
    else:
        verb_flag = ''

    capture_flag = '' if should_capture_output else '-s'
    cmd = f'pytest {capture_flag} {verb_flag} {pytest_ids}'

    os.environ['MODEL_TEMPLATES'] = work_dir
    # Note that just subprocess.run is used instead of run_with_log,
    # since the list of test ids will be printed by pytest
    res = run(cmd, shell=True, check=True, executable="/bin/bash")
    return res

def run_one_domain_tests_already_in_virtualenv(work_dir, all_tests, verbose):
    domains = get_domains_from_tests_list(all_tests)
    if not domains:
        logging.warning('Did not find any tests for the domain')
        return
    if len(domains) > 1:
        raise RuntimeError('The option --run-one-domain-inside-virtual-env may be used for one domain only')
    domain = domains[0]
    if not is_in_virtual_env_in_work_dir(work_dir, domain):
        raise RuntimeError('The option --run-one-domain-inside-virtual-env may be used only'
                           ' inside the virtual environment of the domain')

    print(f'Begin running pytest for domain {domain}', flush=True)
    res = pytest_run_tests(all_tests, work_dir, verbose)
    was_successful = (res.returncode == 0)
    print(f'End running pytest for domain {domain}, was_successful={was_successful}')

    sys_retval = int(not was_successful)
    sys.exit(sys_retval)

def _success_to_str(was_successful):
    return 'OK' if was_successful else 'FAIL'

def rerun_inside_virtual_envs(work_dir, all_tests, args):
    # TODO(LeonidBeynenson): make rerun parametrers be more controllable
    domains = get_domains_from_tests_list(all_tests)
    results = {}
    for domain in domains:
        logging.info(f'Begin running tests for domain {domain}')
        new_argv = ['python3'] + sys.argv
        if not args.domain:
            new_argv.extend(['--domain', domain])
        if not args.workdir:
            new_argv.extend(['--workdir', work_dir])
        new_argv.append('--run-one-domain-inside-virtual-env')

        assert all('"' not in v for v in new_argv), \
                f'Cannot work if arguments contain double quotes:\n{new_argv}'

        cmd = ' '.join(shlex.quote(v) for v in new_argv)
        venv_path = generate_venv_path(work_dir, domain)
        if os.path.isdir(venv_path):
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
    return total_success

def _get_models_path():
    cur_file_path = os.path.abspath(__file__)
    models_path = os.path.join(os.path.dirname(os.path.dirname(cur_file_path)), 'models')

    return models_path

def main():
    # pylint: disable=too-many-branches,too-many-statements
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir',
                        help='The work folder where to instantiate tests and run them. '
                        'Optional. If absent, a temporary folder will be used')
    parser.add_argument('--domain', choices=KNOWN_DOMAIN_FOLDERS, help='Domain name to be tested.  Optional.')
    parser.add_argument('--problem', help='Problem name to be tested. Optional.')
    parser.add_argument('--problem-filter', help='Filter on problem name to be tested. Optional.')
    parser.add_argument('--topic', choices=['train', 'export', 'nncf', 'internal'],
                        help='Topic of tests to be tested. Optional')
    parser.add_argument('--test-id-filter', action='append',
                        help='Filter on test id-s. Optional. Several filters are applied using logical AND')
    parser.add_argument('--verbose', '-v', action='store_true', help='If the tests should be run in verbose mode')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available tests after applying all filters '
                        '(filters like --problem, --topic, --test-id-filter, --test-ids-list-path')
    parser.add_argument('--instantiate-only', action='store_true',
                        help='If the script should instantiate the tests in the work dir only')
    parser.add_argument('--not-instantiate', action='store_true',
                        help='If the script should NOT instantiate the tests in the work dir')
    parser.add_argument('--run-one-domain-inside-virtual-env', action='store_true',
                        help='If the script should run the tests for one domain without work dir instantiation.'
                        ' It is supposed that the script is already run in the proper virtual environment.')
    parser.add_argument('--test-ids-list-path',
                         help='Path to a YAML file with list of test ids that should be executed; '
                         'is applied as a separate filter together with "--domain", "--topic", etc')
    parser.add_argument('--save-list-to-path',
                        help='If --list is used, then save the list of test ids as a JSON struct to the pointed file'
                        '(the struct will be a list, each element of the list will be a dict with fields '
                        '"domain", "problem", "model", "id", "topic"')

    args = parser.parse_args()
    assert not (args.instantiate_only and args.not_instantiate), \
            'Only one of parameters --instantiate-only and --not-instantiate may be set'
    assert not (args.instantiate_only and args.run_one_domain_inside_virtual_env), \
            'Only one of parameters --instantiate-only and --run-one-domain-inside-virtual-env may be set'

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s:%(levelname)s:%(message)s')

    root_path = _get_models_path()
    all_tests, failed_modules = discover_all_tests(root_path, args.domain)
    model_name_to_template = find_all_model_templates(root_path, args.domain)
    fill_template_paths_in_test_elements(all_tests, model_name_to_template)

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
    if args.test_ids_list_path:
        with open(args.test_ids_list_path) as f_t_ids:
            test_ids_list = yaml.safe_load(f_t_ids)
        all_tests = filter_tests_by_many_patterns(all_tests, 'id', test_ids_list)

    logging.info(f'After filtering {len(all_tests)} tests found')

    if args.list:
        if args.save_list_to_path is None:
            print_list_tests(all_tests, args.verbose)
        else:
            save_tests_list(all_tests, args.save_list_to_path)
            logging.info(f'The list of tests is written to the file {args.save_list_to_path}')
        return

    if args.save_list_to_path is not None:
        raise RuntimeError('Command line parameter --save-list-to-path is allowed only if --list parameter is set')

    print('Start working on tests:')
    print_list_tests(all_tests, short=True)

    work_dir = os.path.abspath(args.workdir) if args.workdir else tempfile.mkdtemp(prefix='work_dir_')
    logging.info(f'work_dir = {work_dir}')

    should_instantiate = (not args.run_one_domain_inside_virtual_env) and (not args.not_instantiate)
    if should_instantiate:
        instantiate_work_dir(root_path, work_dir, all_tests, args.verbose)
        logging.info(f'The work_dir {work_dir} is instantiated')

    if args.instantiate_only:
        return

    if args.run_one_domain_inside_virtual_env:
        run_one_domain_tests_already_in_virtualenv(work_dir, all_tests, args.verbose)
        return

    total_success = rerun_inside_virtual_envs(work_dir, all_tests, args)
    if failed_modules:
        failed_modules_str = "  " + "\n  ".join(failed_modules)
        logging.error(f'Loading of the following modules is FAILED:\n'
                      f'{failed_modules_str}')
        total_success = False

    logging.info(f'Total result: {_success_to_str(total_success)}')
    sys_retval = int(not total_success)
    sys.exit(sys_retval)


if __name__ == '__main__':
    main()
