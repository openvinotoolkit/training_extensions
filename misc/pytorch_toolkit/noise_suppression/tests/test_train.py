"""
 Copyright (c) 2021 Intel Corporation

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

        output_dir = mkdtemp()

        #CMD line as in README.txt
        args = [
            "--model_desc=config.json",
            "--output_dir="+output_dir,
            "--num_train_epochs=1",
            "--size_to_read=1",
            "--per_gpu_train_batch_size=1",
            "--total_train_batch_size=8",
            "--dns_datasets=../../../data/noise_suppression/datasets",
            "--logacc=1"
        ]
        main_train(args)

        model_onnx = os.path.join(output_dir,"model.onnx")
        self.assertEqual(True, os.path.exists(model_onnx), model_onnx + " was not created by train script")

        #run convereter to IR as in README.txt
        INTEL_OPENVINO_DIR = os.getenv("INTEL_OPENVINO_DIR")
        export_command = "python {}/deployment_tools/model_optimizer/mo.py".format(INTEL_OPENVINO_DIR)
        export_command += " --input_model " + model_onnx
        export_command += " --output_dir " + output_dir
        res = subprocess.run(export_command, shell=True, check=False)
        self.assertEqual(0, res.returncode, "fail to run "+export_command)

        model_xml = os.path.join(output_dir,"model.xml")
        self.assertEqual(True, os.path.exists(model_xml), model_xml + " was not created by Model Optimizer")

        model_bin = os.path.join(output_dir,"model.bin")
        self.assertEqual(True, os.path.exists(model_bin), model_bin + " was not created by Model Optimizer")

if __name__ == '__main__':
    unittest.main()
