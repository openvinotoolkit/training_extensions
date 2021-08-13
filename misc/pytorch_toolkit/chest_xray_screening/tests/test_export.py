import unittest
import os
import json
from chest_xray_screening.utils.download_weights import download_checkpoint
from chest_xray_screening.utils.exporter import Exporter, OPENVINO_DIR 


def _get_config_():
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path, 'test_config.json'), ) as f1:
        config_file = json.load(f1)

    return config_file['export']

class ExportTest(unittest.TestCase):

    def test_export_onnx(self):
        self.config = _get_config_()
        if not os.path.isdir('model_weights'):
            download_checkpoint()
        self.exporter = Exporter(self.config, optimised=False)
        self.exporter.export_model_onnx()
        self.model_path = self.config['checkpoint']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name_onnx')))

    def test_export_ir(self):

        self.assertTrue(os.path.isdir(OPENVINO_DIR))
        self.config = _get_config_()
        self.exporter = Exporter(self.config,optimised=False)
        self.exporter.export_model_ir()
        self.model_path = self.config['checkpoint']
        self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name')))


    def test_config(self):
        self.config = _get_config_()
        self.model_path = self.config['checkpoint']
        self.input_shape = self.config['input_shape']
        self.output_dir = os.path.split(self.model_path)[0]
        self.assertTrue(self.output_dir)
        self.assertTrue(self.model_path)
        self.assertListEqual(self.input_shape,[1,3,1024,1024])

if __name__ == '__main__':
    unittest.main()
