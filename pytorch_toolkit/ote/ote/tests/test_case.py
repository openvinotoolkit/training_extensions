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

import glob
import json
import logging
import os
import unittest

import mmcv
import torch
import yaml

from ote.tests.utils import collect_ap, run_through_shell
from ote.utils.misc import download_snapshot_if_not_yet


def get_dependencies(template_file):
    output = {}
    with open(template_file) as read_file:
        content = yaml.safe_load(read_file)
        for dependency in content['dependencies']:
            output[dependency['destination'].split('.')[0]] = dependency['source']
        return output


def get_epochs(template_file):
    with open(template_file) as read_file:
        content = yaml.safe_load(read_file)
    return content['hyper_parameters']['basic']['epochs']


def skip_non_instantiated_template_if_its_allowed(template_folder, problem_name, model_name):
    # Note that this is for debug purposes only
    # pylint: disable=no-else-raise
    should_skip_absent_templates = os.environ.get('SHOULD_SKIP_ABSENT_TEMPLATES')
    if not os.path.isdir(template_folder):
        if should_skip_absent_templates:
            raise unittest.SkipTest(f'The template folder for {problem_name}/{model_name}'
                                    f' is not instantiated -- SKIPPING IT')
        else:
            raise unittest.TestCase.failureException(f'The template folder for {problem_name}/{model_name}'
                                                     f' is not instantiated')


def skip_if_cpu_is_not_supported(template_file):
    with open(template_file) as read_file:
        training_targets = [x.lower() for x in yaml.safe_load(read_file)['training_target']]
    if 'cpu' not in training_targets:
        raise unittest.SkipTest('CPU is not supported.')


def create_test_case(domain_name, problem_name, model_name, ann_file, img_root, metric_keys, expected_outputs_dir):
    class TestCaseOteApi(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        topic = 'train'

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.epochs_delta = 1
            cls.total_epochs = get_epochs(cls.template_file) + cls.epochs_delta

            expected_outputs_json = f'{expected_outputs_dir}/{problem_name}/{model_name}.json'

            with open(expected_outputs_json) as read_file:
                cls.expected_outputs = json.load(read_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        def setUp(self):
            self.output_folder = os.path.join(self.template_folder, f'output_{self.id()}')
            os.makedirs(self.output_folder, exist_ok=True)

        def do_evaluation(self, on_gpu):
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            metrics_path = os.path.join(self.output_folder, "metrics.yaml")
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights snapshot.pth'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertLess(abs(self.expected_outputs[metric_key] - value), 1e-4)

        def do_finetuning(self, on_gpu):
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_evaluation_on_gpu(self):
            self.do_evaluation(on_gpu=True)

        def test_evaluation_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_evaluation(on_gpu=False)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_finetuning_on_gpu(self):
            self.do_finetuning(on_gpu=True)

        def test_finetuning_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_finetuning(on_gpu=False)

    return TestCaseOteApi


def create_export_test_case(domain_name, problem_name, model_name, ann_file, img_root, metric_keys, expected_outputs_dir):
    class ExportTestCase(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        topic = 'export'

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.test_export_thr = 3.5

            expected_outputs_json = f'{expected_outputs_dir}/{problem_name}/{model_name}.json'

            with open(expected_outputs_json) as read_file:
                cls.expected_outputs = json.load(read_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

        def setUp(self):
            self.output_folder = os.path.join(self.template_folder, f'output_{self.id()}')
            os.makedirs(self.output_folder, exist_ok=True)

        def do_evaluation(self, export_dir):
            metrics_path = os.path.join(export_dir, "metrics.yaml")
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {metrics_path}'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreaterEqual(value, self.expected_outputs[metric_key] - self.test_export_thr)

        def do_export(self, export_dir, on_gpu):
            if not os.path.exists(export_dir):
                initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
                run_through_shell(
                    f'{initial_command}'
                    f'cd {os.path.dirname(self.template_file)};'
                    f'pip install -r requirements.txt;'
                    f'python export.py'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_export_on_gpu(self):
            export_dir = os.path.join(self.output_folder, 'gpu_export')
            self.do_export(export_dir, on_gpu=True)
            self.do_evaluation(export_dir)

        def test_export_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            export_dir = os.path.join(self.output_folder, 'cpu_export')
            self.do_export(export_dir, on_gpu=False)
            self.do_evaluation(export_dir)

    return ExportTestCase

def create_nncf_test_case(domain_name, problem_name, model_name, ann_file, img_root,
                          template_update_dict,
                          compression_cfg_update_dict=None):
 # pylint: disable=too-many-arguments, too-many-statements
    """
    Note that template_update_dict will be used to update template file
    using the function mmcv.Config.merge_from_dict
    """

    assert template_update_dict, 'Use this function with non-trivial template_update_dict parameter only'
    class NNCFBaseTestCase(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        topic = 'nncf'

        @classmethod
        def setUpClass(cls):
            cls.template_updates_description = cls.generate_template_updates_description(template_update_dict)
            logging.info(f'Begin setting up class for {problem_name}/{model_name}, {cls.template_updates_description}')

            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.src_template_folder = os.path.join(cls.templates_folder,domain_name, problem_name, model_name)

            skip_non_instantiated_template_if_its_allowed(cls.src_template_folder, problem_name, model_name)

            cls.template_folder = cls.generate_template_folder_name(cls.src_template_folder,
                                                                    cls.template_updates_description)
            cls.copy_template_folder(cls.src_template_folder, cls.template_folder)

            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.apply_update_dict_params_to_template_file(cls.template_file,
                                                          template_update_dict,
                                                          compression_cfg_update_dict)

            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )
            logging.info(f'End setting up class for {problem_name}/{model_name}, {cls.template_updates_description}')

        def setUp(self):
            self.output_folder = os.path.join(self.template_folder, f'output_{self.id()}')
            os.makedirs(self.output_folder, exist_ok=True)

        @staticmethod
        def generate_template_updates_description(template_update_dict):
            keys = sorted(template_update_dict.keys())
            template_updates_description = "_".join(k+"="+str(template_update_dict[k]) for k in keys)
            return template_updates_description

        @staticmethod
        def generate_template_folder_name(src_template_folder, template_updates_description):
            assert not src_template_folder.endswith('/')
            template_folder = src_template_folder + '__' +  template_updates_description
            return template_folder

        @staticmethod
        def copy_template_folder(src_template_folder, template_folder):
            logging.info(f'Copying {src_template_folder} to {template_folder}')
            if os.path.isdir(template_folder):
                logging.warning('')
                logging.warning(f'ATTENTION: the folder that should be created for this test case exists!')
                logging.warning(f'           It may cause side effects between tests!')
                logging.warning(f'The folder is `{template_folder}`.\n')
            run_through_shell(f'cp -a "{src_template_folder}" "{template_folder}"')
            assert os.path.isdir(template_folder), f'Cannot create {template_folder}'

        @staticmethod
        def apply_update_dict_params_to_template_file(template_file, template_update_dict, compression_cfg_update_dict):
            template_data = mmcv.Config.fromfile(template_file)
            template_data.dump(template_file + '.backup.yaml')

            if compression_cfg_update_dict:
                assert 'compression.compression_config' not in template_update_dict, (
                    'Config cannot be changed from template_update_dict,'
                    ' if we patch compression config by compression_cfg_update_dict')

                compression_cfg_rel_path = template_data['compression']['compression_config']
                compression_cfg_path = os.path.join(os.path.dirname(template_file), compression_cfg_rel_path)
                new_compression_cfg_path = compression_cfg_path + '.UPDATED_FROM_TEST.json'
                compression_cfg = mmcv.Config.fromfile(compression_cfg_path)
                compression_cfg.merge_from_dict(compression_cfg_update_dict)
                compression_cfg.dump(new_compression_cfg_path)
                assert os.path.isfile(new_compression_cfg_path), f'Cannot write file {new_compression_cfg_path}'

                template_update_dict['compression.compression_config'] = new_compression_cfg_path

            template_data.merge_from_dict(template_update_dict)
            template_data.dump(template_file)

        def do_compress(self):
            log_file = os.path.join(self.output_folder, f'log__{self.id()}.txt')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' | tee {log_file}')
            return log_file

        def do_eval(self, file_to_eval):
            metrics_path = os.path.join(self.output_folder, 'metrics.yaml')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights {file_to_eval}'
                )
            return metrics_path

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_on_gpu(self):
            log_file = self.do_compress()
            return log_file

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_finetune_and_compress_on_gpu(self):
            """
            Note that this training runs a usual training; but since compression flags are
            set inside the template, after several steps of finetuning the train.py script should
            make compression.
            """
            log_file = os.path.join(self.output_folder, f'log__{self.id()}.txt')
            total_epochs = get_epochs(self.template_file)
            total_epochs_with_finetuning = total_epochs + 2
            run_through_shell(
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {total_epochs_with_finetuning}'
                f' | tee {log_file}')
            return log_file

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_eval_on_gpu(self):
            log_file = self.do_compress()

            latest_file = os.path.join(self.output_folder, 'latest.pth')
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            metrics_path = self.do_eval(latest_file)
            return log_file, metrics_path

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_export(self):
            log_file = self.do_compress()

            latest_file = os.path.join(self.output_folder, 'latest.pth')
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python export.py'
                f' --load-weights {latest_file}'
                f' --save-model-to {self.output_folder}'
            )

            model_bin_paths = list(glob.glob(os.path.join(self.output_folder, '*.bin')))
            assert len(model_bin_paths) == 1, (
                f'Wrong result of export.py: globbing "*.bin" in'
                f' {self.output_folder} gives {model_bin_paths}')

            model_bin_path = model_bin_paths[0]

            metrics_path = self.do_eval(model_bin_path)

            return log_file, metrics_path

    return NNCFBaseTestCase
