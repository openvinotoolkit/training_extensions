"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import unittest

from tools.export import ONNXExporter
from tools.utils.get_config import get_config


def create_export_test_case(config_file):
    class TestExport(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(config_file, section='export')
            cls.config = export_config
            cls.exporter = ONNXExporter(cls.config)
            print("test case for config {} created".format(config_file))

        def test_1_encoder_export(self):
            encoder_res_name = self.config.get("res_encoder_name")
            result_model_exists = os.path.exists(encoder_res_name)
            if result_model_exists:
                os.remove(encoder_res_name)
            self.exporter.export_encoder()
            result_model_exists = os.path.exists(encoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_2_decoder_export(self):
            decoder_res_name = self.config.get("res_decoder_name")
            result_model_exists = os.path.exists(decoder_res_name)
            if result_model_exists:
                os.remove(decoder_res_name)
            self.exporter.export_decoder()
            result_model_exists = os.path.exists(decoder_res_name)
            self.assertEqual(True, result_model_exists)

        def test_3_onnx(self):
            metric_onnx = self.exporter.get_onnx_metric()
            target_metric = self.exporter.target_metric
            self.assertGreaterEqual(metric_onnx, target_metric)

        def test_4_encoder_ir_export(self):
            if self.config.get("export_ir"):
                encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'xml')
                result_model_exists = os.path.exists(encoder_res_name)
                if result_model_exists:
                    os.remove(encoder_res_name)
                encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'bin')
                result_model_exists = os.path.exists(encoder_res_name)
                if result_model_exists:
                    os.remove(encoder_res_name)

                self.exporter.export_encoder_ir()
                encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'bin')
                result_model_exists = os.path.exists(encoder_res_name)
                self.assertEqual(True, result_model_exists)
                encoder_res_name = self.config.get("res_encoder_name").replace('onnx', 'xml')
                result_model_exists = os.path.exists(encoder_res_name)
                self.assertEqual(True, result_model_exists)

        def test_5_decoder_ir_export(self):
            if self.config.get("export_ir"):
                decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'xml')
                result_model_exists = os.path.exists(decoder_res_name)
                if result_model_exists:
                    os.remove(decoder_res_name)
                decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'bin')
                result_model_exists = os.path.exists(decoder_res_name)
                if result_model_exists:
                    os.remove(decoder_res_name)

                self.exporter.export_decoder_ir()
                decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'bin')
                result_model_exists = os.path.exists(decoder_res_name)
                self.assertEqual(True, result_model_exists)
                decoder_res_name = self.config.get("res_decoder_name").replace('onnx', 'xml')
                result_model_exists = os.path.exists(decoder_res_name)
                self.assertEqual(True, result_model_exists)

        def test_6_run_ir_model(self):
            if self.config.get("export_ir"):
                ir_metric = self.exporter.get_ir_metric()
                target_metric = self.exporter.target_metric
                self.assertGreaterEqual(ir_metric, target_metric)
    return TestExport


class TestMediumRenderedExport(create_export_test_case("configs/medium_config.yml")):
    "Test case for medium config"


class TestHandwrittenPolynomialsExport(create_export_test_case('configs/polynomials_handwritten_config.yml')):
    "Test case for handwritten polynomials config"


if __name__ == "__main__":
    unittest.main()
