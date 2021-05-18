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

import glob
import json
import logging
import os
import unittest

import mmcv
import yaml

from ote.utils.misc import download_snapshot_if_not_yet, run_through_shell, generate_random_suffix


# these functions contain import-s inside -- this is required for tests discover
def _is_cuda_available():
    import torch
    return torch.cuda.is_available()

def _get_optimisation_config_from_template(*args, **kwargs):
    from ote.modules.compression import get_optimisation_config_from_template
    return get_optimisation_config_from_template(*args, **kwargs)

def _is_optimisation_enabled_in_template(*args, **kwargs):
    from ote.modules.compression import is_optimisation_enabled_in_template
    return is_optimisation_enabled_in_template(*args, **kwargs)

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

def skip_if_cuda_not_available():
    if not _is_cuda_available():
        raise unittest.SkipTest('No GPU found')


def create_test_case(domain_name, problem_name, model_name, ann_file, img_root, metric_keys,
                     expected_outputs_dir, batch_size=1):
    class TestCaseOteApi(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        metrics = metric_keys
        topic = 'train'

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.batch_size = batch_size
            cls.dependencies = get_dependencies(cls.template_file)
            cls.epochs_delta = 1
            cls.total_epochs = get_epochs(cls.template_file) + cls.epochs_delta

            expected_outputs_json = f'{expected_outputs_dir}/{problem_name}/{model_name}.json'

            with open(expected_outputs_json) as read_file:
                cls.expected_outputs = json.load(read_file)

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
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
                f'python3 eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights snapshot.pth'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertLess(abs(self.expected_outputs[metric_key] - value), 1e-3)

        def do_finetuning(self, on_gpu):
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))

        def do_training(self, on_gpu):
            log_file = os.path.join(self.output_folder, 'test_training.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python3 train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size {self.batch_size}'
                f' --epochs 1'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))


        def test_evaluation_on_gpu(self):
            skip_if_cuda_not_available()
            self.do_evaluation(on_gpu=True)

        def test_evaluation_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_evaluation(on_gpu=False)

        def test_finetuning_on_gpu(self):
            skip_if_cuda_not_available()
            self.do_finetuning(on_gpu=True)

        def test_finetuning_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_finetuning(on_gpu=False)

        def test_training_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_training(on_gpu=False)

        def test_smoke_test(self):
            pass

    return TestCaseOteApi


def create_export_test_case(domain_name, problem_name, model_name, ann_file,
                            img_root, metric_keys, expected_outputs_dir):
    class ExportTestCase(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        metrics = metric_keys
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
                f'python3 eval.py'
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
                    f'python3 export.py'
                    f' --load-weights snapshot.pth'
                    f' --save-model-to {export_dir}'
                )

        def test_export_on_gpu(self):
            skip_if_cuda_not_available()
            export_dir = os.path.join(self.output_folder, 'gpu_export')
            self.do_export(export_dir, on_gpu=True)
            self.do_evaluation(export_dir)

        def test_export_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            export_dir = os.path.join(self.output_folder, 'cpu_export')
            self.do_export(export_dir, on_gpu=False)
            self.do_evaluation(export_dir)

        def test_smoke_test(self):
            pass

    return ExportTestCase

def create_nncf_test_case(domain_name, problem_name, model_name, ann_file, img_root,
                          compress_cmd_line_params,
                          template_update_dict=None,
                          compression_cfg_update_dict=None):
 # pylint: disable=too-many-arguments, too-many-statements
    """
    Note that template_update_dict will be used to update template file
    using the function mmcv.Config.merge_from_dict
    """
    if isinstance(compress_cmd_line_params, list):
        compress_cmd_line_params = ' '.join(compress_cmd_line_params)

    assert template_update_dict or compress_cmd_line_params, \
            'Use this function with non-trivial template_update_dict or cmd_line_params_str parameters'

    class NNCFBaseTestCase(unittest.TestCase):
        domain = domain_name
        problem = problem_name
        model = model_name
        topic = 'nncf'

        @classmethod
        def setUpClass(cls):
            cls.compress_cmd_line_params = compress_cmd_line_params
            cls.test_case_description = cls.generate_test_case_description(
                    template_update_dict,
                    compress_cmd_line_params,
                    compression_cfg_update_dict)
            logging.info(f'Begin setting up class for {problem_name}/{model_name}, {cls.test_case_description}')

            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.src_template_folder = os.path.join(cls.templates_folder,domain_name, problem_name, model_name)

            skip_non_instantiated_template_if_its_allowed(cls.src_template_folder, problem_name, model_name)

            src_template_file = os.path.join(cls.src_template_folder, 'template.yaml')
            download_snapshot_if_not_yet(src_template_file, cls.src_template_folder)

            cls.template_folder = cls.generate_template_folder_name(cls.src_template_folder,
                                                                    cls.test_case_description)
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
            )
            logging.info(f'End setting up class for {problem_name}/{model_name}, {cls.test_case_description}')

        def setUp(self):
            self.output_folder = os.path.join(self.template_folder, f'output_{self.id()}')
            os.makedirs(self.output_folder, exist_ok=True)

        @staticmethod
        def generate_test_case_description(template_update_dict,
                                           compress_cmd_line_params,
                                           compression_cfg_update_dict):
            def _dict_to_descr(d):
                if not d:
                    return ''
                return '_'.join(k+'='+str(d[k]) for k in sorted(d.keys()))
            template_updates_description = _dict_to_descr(template_update_dict)

            cmd_line_params = [p.replace('--', '') for p in compress_cmd_line_params.split()]
            cmd_line_params_str = '_'.join(cmd_line_params)

            cfg_update_descr = _dict_to_descr(compression_cfg_update_dict)

            res = '_'.join([template_updates_description,
                            cmd_line_params_str,
                            cfg_update_descr])
            return res

        @staticmethod
        def generate_template_folder_name(src_template_folder, test_case_description):
            assert not src_template_folder.endswith('/')

            src_folder_name = os.path.basename(src_template_folder)
            src_parent_dir = os.path.dirname(src_template_folder)

            random_suffix = generate_random_suffix()
            template_folder_name = src_folder_name + '__' +  random_suffix
            assert len(template_folder_name) < 250
            template_folder_name += '__' + test_case_description

            if len(template_folder_name) > 250:
                # to prevent too long names, may cause *nix error
                template_folder_name = template_folder_name[:250]
                template_folder_name += '---'

            template_folder = os.path.join(src_parent_dir, template_folder_name)
            return template_folder

        @staticmethod
        def copy_template_folder(src_template_folder, template_folder):
            logging.info(f'Copying {src_template_folder} to {template_folder}')
            if os.path.isdir(template_folder):
                logging.warning('')
                logging.warning('ATTENTION: the folder that should be created for this test case exists!')
                logging.warning('           It may cause side effects between tests!')
                logging.warning(f'The folder is `{template_folder}`.\n')
            run_through_shell(f'cp -a --no-target-directory "{src_template_folder}" "{template_folder}"')
            assert os.path.isdir(template_folder), f'Cannot create {template_folder}'

        @staticmethod
        def apply_update_dict_params_to_template_file(template_file, template_update_dict, compression_cfg_update_dict):
            template_data = mmcv.Config.fromfile(template_file)
            template_data.dump(template_file + '.backup.yaml')

            assert _is_optimisation_enabled_in_template(template_data), \
                    f'Template {template_file} does not contain optimisation part'

            if compression_cfg_update_dict:
                compression_cfg_rel_path = _get_optimisation_config_from_template(template_data)
                compression_cfg_path = os.path.join(os.path.dirname(template_file), compression_cfg_rel_path)
                backup_compression_cfg_path = compression_cfg_path + '.BACKUP_FROM_TEST.json'

                compression_cfg = mmcv.Config.fromfile(compression_cfg_path)
                compression_cfg.dump(backup_compression_cfg_path)

                compression_cfg.merge_from_dict(compression_cfg_update_dict)
                compression_cfg.dump(compression_cfg_path)


            template_data.merge_from_dict(template_update_dict)
            template_data.dump(template_file)

        def do_compress(self):
            log_file = os.path.join(self.output_folder, f'log__{self.id()}.txt')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python3 compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                + ' ' + self.compress_cmd_line_params
                + f' | tee {log_file}')
            return log_file

        def do_eval(self, file_to_eval):
            metrics_path = os.path.join(self.output_folder, 'metrics.yaml')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python3 eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights {file_to_eval}'
                )
            return metrics_path

        def test_nncf_compress_on_gpu(self):
            skip_if_cuda_not_available()
            log_file = self.do_compress()
            return log_file

        def test_nncf_compress_and_eval_on_gpu(self):
            skip_if_cuda_not_available()
            log_file = self.do_compress()

            latest_file = os.path.join(self.output_folder, 'latest.pth')
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            metrics_path = self.do_eval(latest_file)
            return log_file, metrics_path

        def test_nncf_compress_and_export(self):
            skip_if_cuda_not_available()
            log_file = self.do_compress()

            latest_file = os.path.join(self.output_folder, 'latest.pth')
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python3 export.py'
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

        def test_smoke_test(self):
            pass

    return NNCFBaseTestCase
