import unittest
import os
import torch

from src.utils.network import Encoder
from src.utils.get_config import get_config
from src.inference import RetrievalInference
from src.utils.exporter import Exporter
from src.utils.downloader import download_checkpoint, download_data



def create_inference_test_for_model():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            export_config = get_config(action = 'evaluation')
            cls.config = export_config
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            gallery_path = parent_dir + export_config['gallery_path']
            query_path = parent_dir + export_config['query_path']
            if not os.path.exists(gallery_path):
                download_data()
            checkpoint = parent_dir + export_config['checkpoint']
            if not os.path.exists(parent_dir + cls.config['checkpoint']):
                download_checkpoint()
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model = Encoder(export_config["zSize"])

            cls.inference = RetrievalInference(
                cls.model, gallery_path, query_path,
                export_config['zSize'], checkpoint,cls.device
            )
            cls.exporter = Exporter(cls.config, openvino=1)

        def test_pytorch_inference(self):
            #gallery = self.inference.gallery_building()
            #map_metric, _, _, _ = self.inference.mean_average_precision(gallery)
            #ndcg_metric, _, _, _ = self.inference.nornalized_discounted_cumulative_gain(gallery)
            map_10, _,_,_, ndcg_10, _,_,_, = self.inference.validate_models(run_type='pytorch')
            self.assertGreaterEqual(map_10, self.config['map_target_metric'])
            self.assertGreaterEqual(ndcg_10, self.config['ndcg_target_metric'])

        def test_onnx_inference(self):
            #model_dir = os.path.split(self.config['checkpoint'])[0]
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            #onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            onnx_checkpoint = parent_dir + self.config.get('model_name_onnx')
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx(parent_dir)
            self.inference.test_onnx(parent_dir + self.config['sample_image_path'], onnx_checkpoint)
            map_10, _,_,_, ndcg_10, _,_,_, = self.inference.validate_models(run_type='onnx',
                                                                        onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(map_10, self.config['map_target_metric'])
            self.assertGreaterEqual(ndcg_10, self.config['ndcg_target_metric'])


        def test_ir_inference(self):
            #model_dir = os.path.split(self.config['checkpoint'])[0]
            current_dir =  os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.abspath(current_dir + "/../")
            #onnx_checkpoint = os.path.join(model_dir, self.config.get('model_name_onnx'))
            #print(onnx_checkpoint)
            onnx_checkpoint = parent_dir + self.config.get('model_name_onnx')
            if not os.path.exists(onnx_checkpoint):
                self.exporter.export_model_onnx(parent_dir)
            map_10, _,_,_, ndcg_10, _,_,_ = self.inference.validate_models(run_type='openvino',
                                                                            onnx_checkpoint=onnx_checkpoint)
            self.assertGreaterEqual(map_10, self.config['map_target_metric'])
            self.assertGreaterEqual(ndcg_10, self.config['ndcg_target_metric'])

        def test_config(self):
            self.config = get_config(action = 'evaluation')
            self.assertEqual(self.config["class_count"], 4)
            self.assertEqual(self.config["zSize"], 48)

    return InferenceTest



class TestInference(create_inference_test_for_model()):
    'Test case for DenseNet121Eff'

if __name__ == '__main__':

    unittest.main()
