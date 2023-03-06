import os
import unittest
from src.utils.get_config import get_config
from src.utils.infer_stage1 import infer_lungseg
from src.utils.infer_stage2 import lungpatch_classifier
from src.utils.downloader import download_data
from src.utils.utils import create_dummy_json_file

def create_inference_test_for_stage1():
    class InferenceTestStage1(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', stage=1)
            if not os.path.exists(cls.config['data_path']):
                download_data()
            if not os.path.exists(cls.config["json_path"]):
                create_dummy_json_file(cls.config["json_path"], stage=1)

        def test_pytorch_inference(self):
            dice = infer_lungseg(self.config, run_type='pytorch')
            self.assertGreater(dice, 0)

        # def test_onnx_inference(self):
        #     dice = infer_lungseg(self.config, run_type='onnx')
        #     self.assertGreater(dice, 0)

        def test_ir_inference(self):
            dice = infer_lungseg(self.config, run_type='ir')
            self.assertGreaterEqual(dice, 0)

    return InferenceTestStage1

def create_inference_test_for_stage2():
    class InferenceTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.config = get_config(action='inference', stage=2)

            if not os.path.exists(cls.config['imgpath']):
                download_data()
            if not os.path.exists(cls.config["jsonpath"]):
                create_dummy_json_file(cls.config["jsonpath"], stage=2)
        def test_pytorch_inference(self):
            accuracy = lungpatch_classifier(self.config, run_type='pytorch')
            self.assertGreater(accuracy, 80)

        def test_onnx_inference(self):
            accuracy = lungpatch_classifier(self.config, run_type='pytorch')
            self.assertGreater(accuracy, 80)

        def test_ir_inference(self):
            accuracy = lungpatch_classifier(self.config, run_type='pytorch')
            self.assertGreater(accuracy, 80)

    return InferenceTest


class TestTrainer(create_inference_test_for_stage1()):
    'Test case for stage1'

class TestTrainerEff(create_inference_test_for_stage2()):
    'Test case for stage2'

if __name__ == '__main__':

    unittest.main()
