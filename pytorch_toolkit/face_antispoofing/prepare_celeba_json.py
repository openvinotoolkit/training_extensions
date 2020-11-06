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

import argparse
import json
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='CelebA_Spoof json preparation')
    parser.add_argument('--root', type=str, default=None, required=True,
                        help='path to root folder of the CelebA_Spoof')
    args = parser.parse_args()
    create_json(mode='train', root_folder=args.root)
    create_json(mode='test', root_folder=args.root)

def create_json(mode, root_folder):
    if mode == 'test':
        list_path = os.path.join(root_folder, 'metas/intra_test/test_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_test.json')
    else:
        assert mode == 'train'
        list_path = os.path.join(root_folder, 'metas/intra_test/train_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_train.json')
    indx=0
    items = {}
    with open('./datasets/small_crops.txt', 'r') as f:
        small_crops = map(lambda x: x.strip(), f.readlines())
        set_ = set(small_crops)
    with open(list_path, 'r') as f:
        data = json.load(f)
        print('Reading dataset info...')
        for indx, path in tqdm(enumerate(data), leave=False):
            labels = data[path] # create list with labels
            bbox_path = os.path.join(root_folder, os.path.splitext(path)[0] + '_BB.txt')
            bbox_f = open(bbox_path, 'r')
            bbox_info = bbox_f.readline().strip().split()[0:4]
            bbox = [int(x) for x in bbox_info] # create bbox with labels
            if len(bbox) < 4 or bbox[2] < 3 or bbox[3] < 3: # filter not existing or too small boxes
                print('Bad bounding box: ', bbox, path)
                continue
            if path in set_:
                print('Bad img cropp: ', path)
            items[indx] = {'path':path, 'labels':labels, 'bbox':bbox}
    with open(save_file, 'w') as f:
        json.dump(items, f)

if __name__ == '__main__':
    main()
