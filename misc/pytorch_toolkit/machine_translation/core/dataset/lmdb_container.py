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
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset


class Page:
    def __init__(self, env, page_size=100000):
        self.env = env
        self.idx = 0
        self.page = {}
        self.page_size = page_size

    def add(self, val, key=None):
        k = self.idx if key is None else key
        self.page[k] = self._encode(val)
        self.idx += 1
        self._check_page_size()

    def close(self):
        if len(self.page):
            self._store_page()

    def _encode(self, s):
        return s.encode(encoding="UTF-8", errors="strict")

    def _check_page_size(self):
        if not (len(self.page) % self.page_size):
            self._store_page()

    def _store_page(self):
        with self.env.begin(write=True) as txn:
            for k, v in self.page.items():
                if not isinstance(k, str):
                    k = self._encode(str(k))
                txn.put(k, v)
            self.page = {}


class LMDBContainer(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(path, lock=False, readonly=True)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
        self.size = self.env.stat()["entries"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        key = self._encode(str(idx))
        val = self._decode(self.cursor.get(key))
        return {"text": val, "key": key}

    def _decode(self, s):
        return s.decode(encoding="UTF-8", errors="strict")

    def _encode(self, s):
        return s.encode(encoding="UTF-8", errors="strict")

    @staticmethod
    def from_txt(names, output_lmdb, page_size=100000, map_size=10**11, is_tqdm=True):
        env = lmdb.open(output_lmdb, map_size=map_size)
        page = Page(env, page_size)
        iterator = tqdm(names) if is_tqdm else names
        for name in iterator:
            for i, _line in enumerate(open(name)):
                line = _line.rstrip("\n")
                if line:
                    page.add(line)
                    if is_tqdm:
                        iterator.set_description(f"[{i}]")
                        iterator.refresh()
        page.close()

    def _get_keys(self):
        keys = [int(key) for key, _ in self.txn.cursor()]
        keys.sort()
        return keys
