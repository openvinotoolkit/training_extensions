import glob
import logging
import os
import sys
from subprocess import run
import tempfile
import unittest

import yaml

ENABLE_TESTS_FOR = {
    'object_detection': [
        'face_detection',
        'person_detection',
        'person_vehicle_bike_detection',
        'text_detection',
        'vehicle_detection',
    ],
    'action_recognition_2': [
        'gesture-recognition',
    ],
}

ENABLE_TRAIN_TESTS = True
ENABLE_EXPORT_TESTS = True
ENABLE_NNCF_TESTS = True

VERBOSE = False

def set_verbosity_from_argv():
    global VERBOSE
    if '-v' in sys.argv or '--verbose' in sys.argv:
        VERBOSE = True

def run_with_log(*args, **kwargs):
    cmd = args[0]
    if VERBOSE:
        logging.info(f'Running command `{cmd}`') #TODO: consider with Ilya
    return run(*args, **kwargs)

class ModelTemplatesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        if VERBOSE:
            logging.info('Running with verbosity=True')
        workdir = os.environ.get('WORKDIR') #TODO: consider with Ilya
        if not workdir:
            cls.work_dir = tempfile.mkdtemp()
        else:
            cls.work_dir = os.path.abspath(workdir)
        templates_pattern_environ = os.environ.get('TEMPLATES_PATTERN') #TODO: consider with Ilya
        if templates_pattern_environ:
            templates_pattern_arg = f'--templates-pattern "{templates_pattern_environ}"'
        else:
            templates_pattern_arg = ''

        if VERBOSE:
            cls.verbosity_flag = '--verbose'
        else:
            cls.verbosity_flag = ''

        run_with_log(f'python3 tools/instantiate.py {templates_pattern_arg} --do-not-load-snapshots {cls.work_dir}', shell=True, check=True)

    def _get_template_files(self):
        #TODO: consider with Ilya
        template_filenames = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
        template_filenames = list(template_filenames)
        return template_filenames

    def test_existence_of_mandatory_files_in_template_dir(self):
        template_files = self._get_template_files()
        for template in template_files:
            template_dirname = os.path.dirname(template)
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'train.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'eval.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'export.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'quantize.py')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'requirements.txt')))
            self.assertTrue(os.path.exists(os.path.join(template_dirname, 'modules.yaml')))

    def test_train_and_eval(self):
        if not ENABLE_TRAIN_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_train_tests.py")}'
                    f' --pattern=train_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

    def test_export(self):
        if not ENABLE_EXPORT_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running export tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_export_tests.py")}'
                    f' --pattern=export_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

    def test_nncf(self):
        # TODO : refactor the class to avoid copying between methods
        if not ENABLE_NNCF_TESTS:
            return

        template_files = self._get_template_files()
        domain_folders = set()
        for template_file in template_files:
            with open(template_file) as read_file:
                domain_folder = yaml.load(read_file, yaml.SafeLoader)['domain'].lower().replace(' ', '_')
                domain_folders.add(domain_folder)

        for domain_folder in domain_folders:
            if domain_folder not in ENABLE_TESTS_FOR:
                continue

            venv_activate_path = os.path.join(self.work_dir, domain_folder, 'venv', 'bin', 'activate')
            for problem_folder in ENABLE_TESTS_FOR[domain_folder]:
                logging.info(f'Running nncf tests for {domain_folder}/{problem_folder}.')
                returncode = run_with_log(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_nncf_tests.py")}'
                    f' --pattern=nncf_tests_{problem_folder}.py {self.verbosity_flag}',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

if __name__ == '__main__':
    set_verbosity_from_argv()
    unittest.main()
