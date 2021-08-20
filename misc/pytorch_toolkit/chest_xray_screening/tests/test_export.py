import unittest
import os
from chest_xray_screening.utils.download_weights import download_checkpoint
from chest_xray_screening.utils.exporter import Exporter, OPENVINO_DIR
from chest_xray_screening.utils.get_config import get_config



def create_export_test_for_densenet121():
    class ExportTest(unittest.TestCase):
        def test_export_onnx(self):
            self.config = get_config(action = 'export')
            if not os.path.isdir('model_weights'):
                download_checkpoint()
            self.exporter = Exporter(self.config, optimised=False)
            self.exporter.export_model_onnx()
            self.model_path = self.config['checkpoint']
            self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name_onnx')))

        def test_export_ir(self):
            self.assertTrue(os.path.isdir(OPENVINO_DIR))
            self.config = get_config(action = 'export')
            if not os.path.isdir('model_weights'):
                download_checkpoint()
            self.exporter = Exporter(self.config, optimised=False)
            self.model_path = self.config['checkpoint']
            if not os.path.exists(os.path.join(self.model_path, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx()
            self.exporter.export_model_ir()
            self.model_path = self.config['checkpoint']
            name_xml = self.config.get('model_name') + '.xml'
            name_bin = self.config.get('model_name') + '.bin'
            xml_status = os.path.exists(os.path.join(self.model_path, name_xml))
            bin_status = os.path.exists(os.path.join(self.model_path, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)



        def test_config(self):
            self.config = get_config(action = 'export')
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1, 3, 1024, 1024])
    return ExportTest

def create_export_test_for_densenet121eff():
    class ExportTestEff(unittest.TestCase):
        def test_export_onnx(self):
            self.config = get_config(action = 'export', optimised = True)
            if not os.path.isdir('model_weights'):
                download_checkpoint()
            self.exporter = Exporter(self.config, optimised=True)
            self.exporter.export_model_onnx()
            self.model_path = self.config['checkpoint']
            self.assertTrue(os.path.join(os.path.split(self.model_path)[0], self.config.get('model_name_onnx')))

        def test_export_ir(self):
            self.assertTrue(os.path.isdir(OPENVINO_DIR))
            self.config = get_config(action = 'export', optimised = True)
            if not os.path.isdir('model_weights'):
                download_checkpoint()
            self.exporter = Exporter(self.config, optimised=True)
            self.model_path = self.config['checkpoint']
            if not os.path.exists(os.path.join(self.model_path, self.config.get('model_name_onnx'))):
                self.exporter.export_model_onnx()
            self.exporter.export_model_ir()
            self.model_path = self.config['checkpoint']
            name_xml = self.config.get('model_name') + '.xml'
            name_bin = self.config.get('model_name') + '.bin'
            xml_status = os.path.exists(os.path.join(self.model_path, name_xml))
            bin_status = os.path.exists(os.path.join(self.model_path, name_bin))
            self.assertTrue(xml_status)
            self.assertTrue(bin_status)


        def test_config(self):
            self.config = get_config(action = 'export', optimised = True)
            self.model_path = self.config['checkpoint']
            self.input_shape = self.config['input_shape']
            self.output_dir = os.path.split(self.model_path)[0]
            self.assertTrue(self.output_dir)
            self.assertTrue(self.model_path)
            self.assertListEqual(self.input_shape, [1, 3, 1024, 1024])
    return ExportTestEff

class TestInference(create_export_test_for_densenet121()):
    'Test case for DenseNet121'

class TestInferenceEff(create_export_test_for_densenet121eff()):
    'Test case for DenseNet121Eff'

if __name__ == '__main__':
    unittest.main()
