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
import io
from torch.utils.data import Dataset
from tqdm import tqdm

class TextContainer(Dataset):
    def __init__(self, corpus):
        self.data = []
        with io.open(corpus, mode='r', encoding='utf-8') as f:
            for line in tqdm(f):
                self.data.append(line.strip())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": self.data[idx],
            "key": idx
        }
