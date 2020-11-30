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

from common.utils import collect_ap, run_through_shell
from ote.utils.misc import download_snapshot_if_not_yet


def get_dependencies(template_file):
    output = {}
    with open(template_file) as read_file:
        content = yaml.load(read_file, yaml.SafeLoader)
        for dependency in content['dependencies']:
            output[dependency['destination'].split('.')[0]] = dependency['source']
        return output


def get_epochs(template_file):
    with open(template_file) as read_file:
        content = yaml.safe_load(read_file)
    return content['hyper_parameters']['basic']['epochs']


def skip_non_instantiated_template_if_its_allowed(template_folder, problem_name, model_name):
    # Note that this is for debug purposes only
    should_skip_absent_templates = os.environ.get('SHOULD_SKIP_ABSENT_TEMPLATES')
    if not os.path.isdir(template_folder):
        if should_skip_absent_templates:
            raise unittest.SkipTest(f'The template folder for {problem_name}/{model_name} is not instantiated -- SKIPPING IT')
        else:
            raise unittest.TestCase.failureException(f'The template folder for {problem_name}/{model_name} is not instantiated')


def skip_if_cpu_is_not_supported(template_file):
    with open(template_file) as read_file:
        training_targets = [x.lower() for x in yaml.safe_load(read_file)['training_target']]
    if 'cpu' not in training_targets:
        raise unittest.SkipTest('CPU is not supported.')


def create_test_case(problem_name, model_name, ann_file, img_root):
    class TestCaseOteApi(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, 'object_detection', problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.epochs_delta = 2
            cls.total_epochs = get_epochs(cls.template_file) + cls.epochs_delta

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_evaluation_on_gpu(self):
            run_through_shell(
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to metrics.yaml'
                f' --load-weights snapshot.pth'
                )

            with open(os.path.join(self.template_folder, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            ap = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'ap'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertLess(abs(content['map'] - ap / 100), 1e-6)

        def test_evaluation_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            run_through_shell(
                'export CUDA_VISIBLE_DEVICES=;'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to metrics.yaml'
                f' --load-weights snapshot.pth'
                )

            with open(os.path.join(self.template_folder, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)

            ap = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == 'ap'][0]

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)
            self.assertLess(abs(content['map'] - ap / 100), 1e-6)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_finetuning_on_gpu(self):
            log_file = os.path.join(self.template_folder, 'test_finetuning.log')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), self.epochs_delta)
            self.assertGreater(ap[-1], 0)

        def test_finetuning_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            log_file = os.path.join(self.template_folder, 'test_finetuning.log')
            run_through_shell(
                'export CUDA_VISIBLE_DEVICES=;'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), self.epochs_delta)
            self.assertGreater(ap[-1], 0)

    return TestCaseOteApi


def create_export_test_case(problem_name, model_name, ann_file, img_root, alt_ssd_export=False):
    class ExportTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, 'object_detection', problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.test_export_thr = 0.031

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

        def do_export(self, folder):
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'pip install -r requirements.txt;'
                f'python export.py'
                f' --load-weights snapshot.pth'
                f' --save-model-to {folder}'
            )

        def export_test_on_gpu(self, alt_ssd_export, thr):
            export_folder = 'gpu_export'
            if not os.path.exists(export_folder):
                self.do_export(export_folder)
            if alt_ssd_export:
                export_dir = os.path.join(self.template_folder, export_folder, 'alt_ssd_export')
            else:
                export_dir = os.path.join(self.template_folder, export_folder)

            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {os.path.join(export_dir, "metrics.yaml")}'
            )

            with open(os.path.join(export_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)
                ap = [metric for metric in content['metrics'] if metric['key'] == 'ap'][0]['value']

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertGreater(ap, content['map'] - thr)

        def export_test_on_cpu(self, alt_ssd_export, thr):
            export_folder = 'cpu_export'
            if not os.path.exists(export_folder):
                self.do_export(export_folder)
            if alt_ssd_export:
                export_dir = os.path.join(self.template_folder, export_folder, 'alt_ssd_export')
            else:
                export_dir = os.path.join(self.template_folder, export_folder)

            run_through_shell(
                f'export CUDA_VISIBLE_DEVICES=;'
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {os.path.join(export_dir, "model.bin")}'
                f' --save-metrics-to {os.path.join(export_dir, "metrics.yaml")}'
            )

            with open(os.path.join(export_dir, "metrics.yaml")) as read_file:
                content = yaml.load(read_file, yaml.SafeLoader)
                ap = [metric for metric in content['metrics'] if metric['key'] == 'ap'][0]['value']

            with open(f'{os.path.dirname(__file__)}/../expected_outputs/{problem_name}/{model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertGreater(ap, content['map'] - thr)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_export_on_gpu(self):
            self.export_test_on_gpu(False, self.test_export_thr)

        def test_export_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.export_test_on_cpu(False, self.test_export_thr)

    class ExportWithAltSsdTestCase(ExportTestCase):

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_alt_ssd_export_on_gpu(self):
            self.export_test_on_gpu(True, self.test_export_thr)

        def test_alt_ssd_export_on_cpu(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.export_test_on_cpu(True, self.test_export_thr)

    if alt_ssd_export:
        return ExportWithAltSsdTestCase

    return ExportTestCase


def create_nncf_test_case(problem_name, model_name, ann_file, img_root, template_update_dict, compression_cfg_update_dict=None):
    """
    Note that template_update_dict will be used to update template file
    using the function mmcv.Config.merge_from_dict
    """

    assert template_update_dict, 'Use this function with non-trivial template_update_dict parameter only'
    class TestCaseOteApi(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.template_updates_description = cls.generate_template_updates_description(template_update_dict)
            logging.info(f'Begin setting up class for {problem_name}/{model_name}, {cls.template_updates_description}')

            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.src_template_folder = os.path.join(cls.templates_folder, 'object_detection', problem_name, model_name)

            skip_non_instantiated_template_if_its_allowed(cls.src_template_folder, problem_name, model_name)

            cls.template_folder = cls.generate_template_folder_name(cls.src_template_folder,
                                                                    cls.template_updates_description)
            cls.copy_template_folder(cls.src_template_folder, cls.template_folder)

            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.apply_update_dict_params_to_template_file(cls.template_file, template_update_dict, compression_cfg_update_dict)

            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)

            # Note that such big threshold is required, since
            # we have very small dataset for training and evaluation:
            # if network compression causes other detections
            # on 2-4 images, the accuracy drop will be significant.
            cls.test_export_thr = 0.05

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )
            logging.info(f'End setting up class for {problem_name}/{model_name}, {cls.template_updates_description}')

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
                compression_cfg_name = os.path.basename(compression_cfg_rel_path)
                compression_cfg_path = os.path.join(os.path.dirname(template_file), compression_cfg_rel_path)
                new_compression_cfg_path = compression_cfg_path + '.UPDATED_FROM_TEST.json'
                compression_cfg = mmcv.Config.fromfile(compression_cfg_path)
                compression_cfg.merge_from_dict(compression_cfg_update_dict)
                compression_cfg.dump(new_compression_cfg_path)
                assert os.path.isfile(new_compression_cfg_path), f'Cannot write file {new_compression_cfg_path}'

                template_update_dict['compression.compression_config'] = new_compression_cfg_path

            template_data.merge_from_dict(template_update_dict)
            template_data.dump(template_file)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_on_gpu(self):
            log_file = os.path.join(self.template_folder, f'log__{self.id()}.txt')
            run_through_shell(
                f'cd {self.template_folder};'
                f'python compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {self.template_folder}/output_{self.id()}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' | tee {log_file}')

            ap = collect_ap(log_file)
            self.assertGreater(ap[-1], 0)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_finetune_and_compress_on_gpu(self):
            log_file = os.path.join(self.template_folder, f'log__{self.id()}.txt')
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
                f' --save-checkpoints-to {self.template_folder}/output_{self.id()}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --epochs {total_epochs_with_finetuning}'
                f' | tee {log_file}')

            ap = collect_ap(log_file)
            self.assertGreater(ap[-1], 0)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_eval_on_gpu(self):
            log_file = os.path.join(self.template_folder, f'log__{self.id()}.txt')
            checkpoints_dir = f'{self.template_folder}/output_{self.id()}'
            run_through_shell(
                f'cd {self.template_folder};'
                f'python compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {checkpoints_dir}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' | tee {log_file}')
            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]
            logging.info(f'From training last_compress_ap={last_compress_ap}')

            latest_file = f'{checkpoints_dir}/latest.pth'
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            metrics_path = f'{checkpoints_dir}/metrics.yaml'
            run_through_shell(
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --load-weights {latest_file}'
                )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            ap = [metric['value'] for metric in content['metrics'] if metric['key'] == 'ap'][0]
            ap = ap/100

            logging.info(f'Evaluation result ap={ap}')
            self.assertLess(abs(last_compress_ap - ap), 1e-6)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_export(self):
            log_file = os.path.join(self.template_folder, f'log__{self.id()}.txt')
            checkpoints_dir = f'{self.template_folder}/output_{self.id()}'
            run_through_shell(
                f'cd {self.template_folder};'
                f'python compress.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --load-weights snapshot.pth'
                f' --save-checkpoints-to {checkpoints_dir}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' | tee {log_file}')
            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]

            latest_file = f'{checkpoints_dir}/latest.pth'
            self.assertTrue(os.path.isfile(latest_file), f'Cannot find the latest.pth in path `{latest_file}`')

            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python export.py'
                f' --load-weights {latest_file}'
                f' --save-model-to {checkpoints_dir}'
            )

            model_bin_paths = list(glob.glob(os.path.join(checkpoints_dir, '*.bin')))
            assert len(model_bin_paths) == 1, (
                    f'Wrong result of export.py: globbing "*.bin" in'
                    f' {checkpoints_dir} gives {model_bin_paths}')
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {ann_file}'
                f' --test-data-roots {img_root}'
                f' --load-weights {model_bin_paths[0]}'
                f' --save-metrics-to {os.path.join(checkpoints_dir, "metrics.yaml")}'
            )

            with open(os.path.join(checkpoints_dir, "metrics.yaml")) as read_file:
                content = yaml.safe_load(read_file)
                ap = [metric for metric in content['metrics'] if metric['key'] == 'ap'][0]['value']
                ap = ap/100

            logging.info(f'From training last_compress_ap={last_compress_ap}')
            logging.info(f'From evaluation of OpenVINO(TM) model ap={ap}')
            self.assertGreater(ap, last_compress_ap - self.test_export_thr)

    return TestCaseOteApi

