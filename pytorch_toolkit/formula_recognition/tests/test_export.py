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

import unittest

import numpy as np
import yaml

from tools.export import ONNXExporter


class TestExport(unittest.TestCase):
    def setUp(self):
        with open('configs/export_config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = config
        self.exporter = ONNXExporter(config)

    def test_1_encoder_export(self):
        try:
            self.exporter.export_encoder()
        except:
            self.fail("Exception raised while exporting encoder to ONNX")

    def test_2_decoder_export(self):
        try:
            features = self.exporter.run_encoder()
            self.exporter.export_decoder(*features)
        except:
            self.fail("Exception raised while exporting decoder to ONNX")

    def test_3_onnx(self):
        row_enc_out, h, c, O_t = self.exporter.run_encoder()
        pred_onnx = self.exporter.vocab.construct_phrase(
            self.exporter.run_decoder(h, c, O_t, row_enc_out).astype(np.int32))
        _, targets = self.exporter.model(self.exporter.img)
        pred_pytorch = self.exporter.vocab.construct_phrase(targets[0])
        self.assertEqual(pred_onnx, pred_pytorch)

    def test_4_encoder_ir_export(self):
        if self.config.get("export_ir"):
            try:
                self.exporter.export_encoder_ir()
            except:
                self.fail("Exception raised while exporting encoder to openvino IR")

    def test_5_decoder_ir_export(self):
        if self.config.get("export_ir"):
            try:
                self.exporter.export_decoder_ir()
            except:
                self.fail("Exception raised while exporting decoder to openvino IR")

    def test_6_run_ir_model(self):
        if self.config.get("export_ir"):
            ir_pred = self.exporter.vocab.construct_phrase(self.exporter.run_ir_model())
            _, targets = self.exporter.model(self.exporter.img)
            pred_pytorch = self.exporter.vocab.construct_phrase(targets[0])
            self.assertEqual(ir_pred, pred_pytorch)


if __name__ == "__main__":
    unittest.main()
