import unittest
from tools.export import ONNXExporter
import yaml
import numpy as np


class TestExport(unittest.TestCase):
    def setUp(self):
        with open('configs/export_config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.exporter = ONNXExporter(config)

    def test_encoder_export(self):
        self.assertTrue(self.exporter.export_encoder())

    def test_decoder_export(self):
        features = self.exporter.run_encoder()
        self.assertTrue(self.exporter.export_decoder(*features))

    def test_onnx(self):
        row_enc_out, h, c, O_t = self.exporter.run_encoder()
        pred_onnx = self.exporter.vocab.construct_phrase(
            self.exporter.run_decoder(h, c, O_t, row_enc_out).astype(np.int32))
        _, targets = self.exporter.model(self.exporter.img)
        pred_pytorch = self.exporter.vocab.construct_phrase(targets[0])
        self.assertEqual(pred_onnx, pred_pytorch)


if __name__ == "__main__":
    unittest.main()
