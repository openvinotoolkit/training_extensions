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

from torch.utils.data import Dataset
from PIL import Image
import os
import os.path as osp


class EyeDataset(Dataset):
    def __init__(self, root_folder, mode='train', transformations=None):
        self.transformations = transformations
        self.data = []
        for subdir, dirs, files in os.walk(osp.join(root_folder, mode)):
            for i, file in enumerate(files):                
                full_path = os.path.join(subdir, file)
                state = 1 if file[0] == 'o' else 0
                if mode == 'train':
                    self.data.append({'filename': full_path, 'label': state})
                else:
                    self.data.append({'filename': full_path, 'label': state})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['filename']).convert('RGB')
        if self.transformations is not None:
            img = self.transformations(img)
        return img, item['label'], item['filename']
