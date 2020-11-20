import glob
import logging
import os
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


class ModelTemplatesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.work_dir = tempfile.mkdtemp()
        run(f'python3 tools/instantiate.py --do-not-load-snapshots {cls.work_dir}', shell=True, check=True)

    def test_existence_of_mandatory_files_in_template_dir(self):
        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
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

        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
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
                returncode = run(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_train_tests.py")}'
                    f' --pattern=train_tests_{problem_folder}.py',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)

    def test_export(self):
        if not ENABLE_EXPORT_TESTS:
            return

        template_files = glob.glob(f'{self.work_dir}/**/template.yaml', recursive=True)
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
                returncode = run(
                    f'. {venv_activate_path};'
                    f'export MODEL_TEMPLATES={self.work_dir};'
                    f'python3 {os.path.join(domain_folder, "tests", "run_export_tests.py")}'
                    f' --pattern=export_tests_{problem_folder}.py',
                    shell=True,
                    check=True,
                    executable="/bin/bash").returncode
                self.assertEqual(returncode, 0)


if __name__ == '__main__':
    unittest.main()
