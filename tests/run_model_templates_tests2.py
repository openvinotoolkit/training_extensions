import argparse
import logging
import os
import tempfile
from subprocess import run


def parse_args():
    domains = os.listdir('models')
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', choices=domains)
    parser.add_argument('--work_dir')
    parser.add_argument('--test',
                        help='It is possible to specify test or test case directly: '
                             'models/text_spotting/tests/train_tests_alphanumeric_text_spotting.py::TextSpotting0003TestCase::test_evaluation_on_gpu')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.test is not None:
        assert args.domains is None
        assert args.test.startswith('models/')
        domains = [args.test.split('/')[1]]
        to_test = args.test
    else:
        if args.domains is None:
            domains = os.listdir('models')
        to_test = None

    if args.work_dir is not None:
        destination = args.work_dir
    else:
        destination = tempfile.mkdtemp()

    for domain in domains:
        venv_path = os.path.join(destination, domain, 'venv')
        if not os.path.exists(venv_path):
            run(f'python3 tools/instantiate.py {destination} --templates-filter models/{domain}/**/template.yaml --verbose',
                check=True, shell=True)
        else:
            logging.warning(f'You are running tests inside previously created virtual environments. They might be outdated! {venv_path}')

    for domain in domains:
        if args.test is None:
            to_test = f'models/{domain}/tests/*'
        run(f'source {destination}/{domain}/venv/bin/activate && '
            f'export MODEL_TEMPLATES={destination} && '
            f'pytest {to_test} -v',
            check=True, shell=True, executable='/bin/bash')
