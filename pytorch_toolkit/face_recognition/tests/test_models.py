"""
 Copyright (c) 2018 Intel Corporation
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
import os
import torch

from model.common import models_backbones, models_landmarks
from utils.utils import save_model_cpu, load_model_state


class BackbonesTests(unittest.TestCase):
    """Tests for backbones"""
    def test_output_shape(self):
        """Checks output shape"""
        embed_size = 256
        for model_type in models_backbones.values():
            model = model_type(embedding_size=embed_size, feature=True).eval()
            batch = torch.Tensor(1, 3, *model.get_input_res()).uniform_()
            output = model(batch)
            self.assertEqual(list(output.shape), list((1, embed_size, 1, 1)))

    def test_save_load_snap(self):
        """Checks an ability to save and load model correctly"""
        embed_size = 256
        snap_name = os.path.join(os.getcwd(), 'test_snap.pt')
        for model_type in models_backbones.values():
            model = model_type(embedding_size=embed_size, feature=True).eval()
            batch = torch.Tensor(1, 3, *model.get_input_res()).uniform_()
            output = model(batch)
            save_model_cpu(model, None, snap_name, 0, write_solverstate=False)

            model_loaded = model_type(embedding_size=embed_size, feature=True)
            load_model_state(model_loaded, snap_name, -1, eval_state=True)

            output_loaded = model_loaded(batch)

            self.assertEqual(torch.norm(output - output_loaded), 0)


class LandnetTests(unittest.TestCase):
    """Tests for landmark regressor"""
    def test_output_shape(self):
        """Checks output shape"""
        model = models_landmarks['landnet']().eval()
        batch = torch.Tensor(1, 3, *model.get_input_res())
        output = model(batch)
        self.assertEqual(list(output.shape), list((1, 10, 1, 1)))

    def test_save_load_snap(self):
        """Checks an ability to save and load model correctly"""
        snap_name = os.path.join(os.getcwd(), 'test_snap.pt')
        model = models_landmarks['landnet']().eval()
        batch = torch.Tensor(1, 3, *model.get_input_res()).uniform_()
        output = model(batch)
        save_model_cpu(model, None, snap_name, 0, write_solverstate=False)

        model_loaded = models_landmarks['landnet']()
        load_model_state(model_loaded, snap_name, -1, eval_state=True)

        output_loaded = model_loaded(batch)

        self.assertEqual(torch.norm(output - output_loaded), 0)

if __name__ == '__main__':
    unittest.main()
