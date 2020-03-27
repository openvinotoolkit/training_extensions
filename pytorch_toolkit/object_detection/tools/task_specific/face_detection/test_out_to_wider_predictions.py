import argparse
import os
from tqdm import tqdm

import mmcv
import torch

from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='This script converts output of test.py (mmdetection) to '
                                                 'a set of files that can be passed to official WiderFace '
                                                 'evaluation procedure.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('input', help='output result file from test.py')
    parser.add_argument('out_folder', help='folder where to store WiderFace '
                                           'evaluation-friendly output')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.input is not None and not args.input.endswith(('.pkl', '.pickle')):
        raise ValueError('The input file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(args.input)

    wider_friendly_results = []
    for i, sample in enumerate(tqdm(dataset)):
        filename = sample['img_meta'][0]._data['filename']
        folder, image_name = filename.split('/')[-2:]
        wider_friendly_results.append({'folder': folder, 'name': image_name[:-4],
                        'boxes': results[i][0]})

    for result in wider_friendly_results:
        folder = os.path.join(args.out_folder,  result['folder'])
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, result['name'] + '.txt'), 'w') as f:
            f.write(result['name'] + '\n')
            f.write(str(len(result['boxes'])) + '\n')
            for b in result['boxes']:
                b = b[0], b[1], b[2] - b[0], b[3] - b[1], b[4]
                f.write(' '.join([str(x) for x in b]) + '\n')

if __name__ == '__main__':
    main()
