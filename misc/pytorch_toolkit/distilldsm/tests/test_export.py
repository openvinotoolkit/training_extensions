import unittest
import os
from src.utils.utils import load_json
from src.utils.download_weights import download_checkpoint
from src.utils.exporter import Exporter


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_json("configs/hearts/heart_config.json")
        cls.config["model_name"] = "Distill"
        cls.model_filename = "model_weights/distill_dsm.h5"
        if not os.path.isdir('model_weights'):
            download_checkpoint()
        cls.config["model_onnx_filename"] = cls.model_filename.split('.')[0]+".onnx"
        cls.exporter = Exporter(cls.config)

    def test_export_onnx(self):
        self.exporter.export_model_onnx()
        self.assertTrue(os.path.exists(self.config.get('model_onnx_filename')))

    def test_export_ir(self):
        model_dir = self.config.get('model_onnx_filename')
        if not os.path.exists(model_dir):
            self.exporter.export_model_onnx()
        self.exporter.export_model_ir()
        name_xml = self.config['model_name'] + '.xml'
        name_bin = self.config['model_name'] + '.bin'
        xml_status = os.path.exists(os.path.join(model_dir, name_xml))
        bin_status = os.path.exists(os.path.join(model_dir, name_bin))
        self.assertTrue(xml_status)
        self.assertTrue(bin_status)
