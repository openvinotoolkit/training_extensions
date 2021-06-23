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
import subprocess
import unittest
from tempfile import mkdtemp
from train import main as main_train



class TestTrain(unittest.TestCase):
    def test_train(self):

        with self.assertRaises(Exception):
            output_dir = mkdtemp()

            #CMD line from README.txt
            args = "--model_desc=config.json \
            --output_dir={} \
            --num_train_epochs=1 \
            --size_to_read=1 \
            --per_gpu_train_batch_size=1 \
            --total_train_batch_size=8 \
            --dns_datasets=../../../data/noise_suppression/datasets".format(output_dir).split()

            main_train(args)

            model_onnx = os.path.join(output_dir,"model.onnx")
            self.assertEqual(True, os.path.exists(model_onnx))

            #run convereter to IR
            INTEL_OPENVINO_DIR = os.getenv("INTEL_OPENVINO_DIR")
            export_command = "{}/bin/setupvars.sh && \
            python {}/deployment_tools/model_optimizer/mo.py \
            --input_model {} \
            --output_dir {}".format(INTEL_OPENVINO_DIR, INTEL_OPENVINO_DIR, model_onnx, output_dir)
            subprocess.run(export_command,shell=True, check=True)

            model_xml = os.path.join(output_dir,"model.xml")
            self.assertEqual(True, os.path.exists(model_xml))

            model_bin = os.path.join(output_dir,"model.bin")
            self.assertEqual(True, os.path.exists(model_bin))

if __name__ == '__main__':
    unittest.main()
