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
import logging
import os
import unittest

import mmcv
import torch
import yaml

from ote.utils.misc import download_snapshot_if_not_yet
from ote.tests.test_case import (create_export_test_case,
                                 create_test_case,
                                 get_dependencies,
                                 get_epochs,
                                 skip_non_instantiated_template_if_its_allowed)
from ote.tests.utils import collect_ap, run_through_shell


def create_object_detection_test_case(**kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    return create_test_case('object_detection',
                            **kwargs,
                            metric_keys=['bbox'],
                            expected_outputs_dir=expected_outputs_dir)


def create_object_detection_export_test_case(alt_ssd_export=False, **kwargs):
    expected_outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'expected_outputs')
    ExportTestCase = create_export_test_case('object_detection',
                                             **kwargs,
                                             metric_keys=['bbox'],
                                             expected_outputs_dir=expected_outputs_dir)
    if alt_ssd_export:
        class ExportWithAltSsdTestCase(ExportTestCase):

            @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
            def test_alt_ssd_export_on_gpu(self):
                export_dir = os.path.join(self.template_folder, 'gpu_export')
                self.do_export(export_dir, on_gpu=True)
                export_dir = os.path.join(export_dir, 'alt_ssd_export')
                self.do_evaluation(export_dir)

            def test_alt_ssd_export_on_cpu(self):
                export_dir = os.path.join(self.template_folder, 'gpu_export')
                self.do_export(export_dir, on_gpu=True)
                export_dir = os.path.join(export_dir, 'alt_ssd_export')
                self.do_evaluation(export_dir)

        return ExportWithAltSsdTestCase

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

def create_object_detection_nncf_test_case(problem_name, model_name, ann_file, img_root,
                                           template_update_dict,
                                           compression_cfg_update_dict=None,
                                           test_export_threshold=0.09):

    NNCFBaseTestCase = create_nncf_test_case('object_detection', problem_name, model_name, ann_file, img_root,
                                             template_update_dict=template_update_dict,
                                             compression_cfg_update_dict=compression_cfg_update_dict)

    class NNCFObjectDetectionTestCase(NNCFBaseTestCase):
        def setUp(self):
            super().setUp()

            # Note that such big threshold is required, since
            # we have very small dataset for training and evaluation:
            # if network compression causes other detections
            # on 2-4 images, the accuracy drop will be significant.
            self.test_export_thr = test_export_threshold

        @staticmethod
        def _get_bbox_metric(metrics_path):
            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            ap = [metric['value'] for metric in content['metrics'] if metric['key'] == 'bbox'][0]
            ap = ap/100
            return ap

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_on_gpu(self):
            log_file = super().test_nncf_compress_on_gpu()
            ap = collect_ap(log_file)
            self.assertGreater(ap[-1], 0)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_finetune_and_compress_on_gpu(self):
            log_file = super().test_nncf_finetune_and_compress_on_gpu()
            ap = collect_ap(log_file)
            self.assertGreater(ap[-1], 0)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_eval_on_gpu(self):
            log_file, metrics_path = super().test_nncf_compress_and_eval_on_gpu()

            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]
            logging.info(f'From training last_compress_ap={last_compress_ap}')

            ap = self._get_bbox_metric(metrics_path)
            logging.info(f'Evaluation result ap={ap}')
            self.assertLess(abs(last_compress_ap - ap), 1e-6)

            return log_file, metrics_path

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_nncf_compress_and_export(self):
            log_file, metrics_path = super().test_nncf_compress_and_export()

            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]
            logging.info(f'From training last_compress_ap={last_compress_ap}')

            ap = self._get_bbox_metric(metrics_path)
            logging.info(f'Evaluation after export result ap={ap}')

            self.assertGreater(ap, last_compress_ap - self.test_export_thr)

    return NNCFObjectDetectionTestCase
