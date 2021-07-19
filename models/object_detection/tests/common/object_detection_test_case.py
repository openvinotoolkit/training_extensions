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

import logging
import os

import yaml

from ote.tests.test_case import (create_export_test_case,
                                 create_nncf_test_case,
                                 create_test_case,
                                 skip_if_cuda_not_available)
from ote.tests.utils import collect_ap


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

            def test_alt_ssd_export_on_gpu(self):
                skip_if_cuda_not_available()
                export_dir = os.path.join(self.output_folder, 'gpu_export')
                self.do_export(export_dir, on_gpu=True)
                export_dir = os.path.join(export_dir, 'alt_ssd_export')
                self.do_evaluation(export_dir)

            def test_alt_ssd_export_on_cpu(self):
                export_dir = os.path.join(self.output_folder, 'cpu_export')
                self.do_export(export_dir, on_gpu=True)
                export_dir = os.path.join(export_dir, 'alt_ssd_export')
                self.do_evaluation(export_dir)

        return ExportWithAltSsdTestCase

    return ExportTestCase


def create_object_detection_nncf_test_case(problem_name, model_name, ann_file, img_root,
                                           compression_cmd_line_parameters,
                                           template_update_dict=None,
                                           compression_cfg_update_dict=None,
                                           test_export_threshold=0.09):
    # pylint: disable=too-many-arguments
    if template_update_dict is None:
        template_update_dict = {}

    NNCFBaseTestCase = create_nncf_test_case('object_detection', problem_name, model_name, ann_file, img_root,
                                             compression_cmd_line_parameters,
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
            ap = ap / 100
            return ap

        def test_nncf_compress_on_gpu(self):
            skip_if_cuda_not_available()
            log_file = super().test_nncf_compress_on_gpu()
            ap = collect_ap(log_file)
            self.assertGreaterEqual(ap[-1], 0)

        def test_nncf_compress_and_eval_on_gpu(self):
            skip_if_cuda_not_available()
            log_file, metrics_path = super().test_nncf_compress_and_eval_on_gpu()

            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]
            logging.info(f'From training last_compress_ap={last_compress_ap}')

            ap = self._get_bbox_metric(metrics_path)
            logging.info(f'Evaluation result ap={ap}')
            self.assertLess(abs(last_compress_ap - ap), 1e-6)

            return log_file, metrics_path

        def test_nncf_compress_and_export(self):
            skip_if_cuda_not_available()
            log_file, metrics_path = super().test_nncf_compress_and_export()

            compress_ap = collect_ap(log_file)
            last_compress_ap = compress_ap[-1]
            logging.info(f'From training last_compress_ap={last_compress_ap}')

            ap = self._get_bbox_metric(metrics_path)
            logging.info(f'Evaluation after export result ap={ap}')

            self.assertGreater(ap, last_compress_ap - self.test_export_thr)

    return NNCFObjectDetectionTestCase
