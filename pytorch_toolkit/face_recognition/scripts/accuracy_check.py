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

import argparse
import glog as log

import numpy as np
import torch
from tqdm import tqdm
import cv2 as cv

from utils.utils import load_model_state
from utils.ie_tools import load_ie_model
from model.common import models_backbones, models_landmarks

def main():
    """Runs the accuracy check"""
    parser = argparse.ArgumentParser(description='Accuracy check script (pt vs caffe)')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of the face embedding.')
    parser.add_argument('--snap', type=str, required=True, help='Snapshot to convert.')
    parser.add_argument('--device', '-d', default=0, type=int, help='Device for model placement.')
    parser.add_argument('--model', choices=list(models_backbones.keys()) + list(models_landmarks.keys()), type=str,
                        default='rmnet')

    # IE-related options
    parser.add_argument('--ie_model', type=str, required=True)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d_ie", "--device_ie",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)

    args = parser.parse_args()

    max_err = 0.
    with torch.cuda.device(args.device):
        if args.model in models_landmarks.keys():
            pt_model = models_landmarks[args.model]
        else:
            pt_model = models_backbones[args.model](embedding_size=args.embed_size, feature=True)
        pt_model = load_model_state(pt_model, args.snap, args.device)

        ie_model = load_ie_model(args.ie_model, args.device_ie, args.plugin_dir, args.cpu_extension)
        np.random.seed(0)

        for _ in tqdm(range(100)):
            input_img = np.random.randint(0, high=255, size=(*pt_model.get_input_res(), 3), dtype=np.uint8)
            input_bgr = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)

            input_pt = torch.unsqueeze(torch.from_numpy(input_img.transpose(2, 0, 1).astype('float32') / 255.).cuda(),
                                       dim=0)
            pt_output = (pt_model(input_pt)).data.cpu().numpy().reshape(1, -1)
            ie_output = ie_model.forward(input_bgr).reshape(1, -1)

            max_err = max(np.linalg.norm(pt_output - ie_output, np.inf), max_err)

    log.info('Max l_inf error: %e', max_err)

if __name__ == '__main__':
    main()
