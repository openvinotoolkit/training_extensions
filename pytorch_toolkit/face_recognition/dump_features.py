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

import sys
import argparse
import os
import os.path as osp

from tqdm import tqdm
import numpy as np
import glog as log
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as t

from scripts.matio import save_mat
from model.common import models_backbones
from datasets.megaface import MegaFace
from datasets.trillion_pairs import TrillionPairs
from utils.utils import load_model_state
from utils.augmentation import ResizeNumpy, NumpyToTensor


def clean_megaface(filenames, features, noises_list_path):
    """Filters megaface from outliers"""
    with open(noises_list_path, 'r') as f:
        noises_list = f.readlines()
        noises_list = [line.strip() for line in noises_list]
    clean_features = np.zeros((features.shape[0], features.shape[1] + 1), dtype=np.float32)

    for i, filename in enumerate(tqdm(filenames)):
        clean_features[i, 0: features.shape[1]] = features[i, :]
        for line in noises_list:
            if line in filename:
                clean_features[i, features.shape[1]] = 100.0
                break

    return clean_features


def clean_facescrub(filenames, features, noises_list_path):
    """Replaces wrong instances of identities from the Facescrub with the centroids of these identities"""
    clean_feature_size = features.shape[1] + 1
    with open(noises_list_path, 'r') as f:
        noises_list = f.readlines()
        noises_list = [osp.splitext(line.strip())[0] for line in noises_list]
    clean_features = np.zeros((features.shape[0], clean_feature_size), dtype=np.float32)

    centroids = {}
    for i, filename in enumerate(tqdm(filenames)):
        clean_features[i, 0: features.shape[1]] = features[i, :]
        id_name = osp.basename(filename).split('_')[0]
        if not id_name in centroids:
            centroids[id_name] = np.zeros(clean_feature_size, dtype=np.float32)
        centroids[id_name] += clean_features[i, :]

    for i, file_path in enumerate(tqdm(filenames)):
        filename = osp.basename(file_path)
        for line in noises_list:
            if line in filename.replace(' ', '_'):
                id_name = filename.split('_')[0]
                clean_features[i, :] = centroids[id_name] + np.random.uniform(-0.001, 0.001, clean_feature_size)
                clean_features[i, :] /= np.linalg.norm(clean_features[i, :])
                break

    return clean_features


@torch.no_grad()
def main(args):
    input_filenames = []
    output_filenames = []
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not args.trillion_format:
        log.info('Reading info...')
        with open(os.path.join(args.input_dir, os.path.basename(args.input_list)), 'r') as f:
            lines = f.readlines()

            for line in tqdm(lines):
                info = line.strip().split('|')
                file = info[0].strip()
                filename = os.path.join(input_dir, file)

                path, _ = osp.split(filename)
                out_folder = path.replace(input_dir, output_dir)
                if not osp.isdir(out_folder):
                    os.makedirs(out_folder)

                landmarks = None
                bbox = None

                if len(info) > 2:
                    landmarks = info[1].strip().split(' ')
                    landmarks = [float(x) for x in landmarks]
                    bbox = info[2].strip().split(' ')
                    bbox = [int(float(x)) for x in bbox]
                outname = filename.replace(input_dir, output_dir) + args.file_ending
                input_filenames.append({'path': filename, 'landmarks': landmarks, 'bbox': bbox})
                output_filenames += [outname]

        nrof_images = len(input_filenames)
        log.info("Total number of images: ", nrof_images)
        dataset = MegaFace(input_filenames)
    else:
        dataset = TrillionPairs(args.input_dir, osp.join(args.input_dir, 'testdata_lmk.txt'), test_mode=True)
        nrof_images = len(dataset)

    emb_array = np.zeros((nrof_images, args.embedding_size), dtype=np.float32)

    dataset.transform = t.Compose([ResizeNumpy(models_backbones[args.model].get_input_res()),
                                   NumpyToTensor(switch_rb=True)])
    val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=5, shuffle=False)

    model = models_backbones[args.model](embedding_size=args.embedding_size, feature=True)
    assert args.snap is not None
    log.info('Snapshot ' + args.snap + ' ...')
    log.info('Extracting embeddings ...')
    model = load_model_state(model, args.snap, args.devices[0], eval_state=True)
    model = torch.nn.DataParallel(model, device_ids=args.devices, output_device=args.devices[0])

    f_output_filenames = []

    with torch.cuda.device(args.devices[0]):
        for i, data in enumerate(tqdm(val_loader), 0):
            idxs, imgs = data['idx'], data['img']
            batch_embeddings = F.normalize(model(imgs), p=2, dim=1).data.cpu().numpy()
            batch_embeddings = batch_embeddings.reshape(batch_embeddings.shape[0], -1)
            path_indices = idxs.data.cpu().numpy()

            start_index = i*args.batch_size
            end_index = min((i+1)*args.batch_size, nrof_images)
            assert start_index == path_indices[0]
            assert end_index == path_indices[-1] + 1
            assert emb_array[start_index:end_index, :].shape == batch_embeddings.shape
            emb_array[start_index:end_index, :] = batch_embeddings

            if not args.trillion_format:
                for index in path_indices:
                    f_output_filenames.append(output_filenames[index])

    assert len(output_filenames) == len(output_filenames)

    log.info('Extracting features Done.')

    if args.trillion_format:
        save_mat(args.file_ending, emb_array)
    else:
        if 'megaface_noises.txt' in args.noises_list:
            log.info('Cleaning Megaface features')
            emb_array = clean_megaface(f_output_filenames, emb_array, args.noises_list)
        elif 'facescrub_noises.txt' in args.noises_list:
            log.info('Cleaning Facescrub features')
            emb_array = clean_facescrub(f_output_filenames, emb_array, args.noises_list)
        else:
            log.info('Megaface features are not cleaned up.')

        log.info('Saving features to files...')
        for i in tqdm(range(len(f_output_filenames))):
            save_mat(f_output_filenames[i], emb_array[i, :])


def parse_argument(argv):
    parser = argparse.ArgumentParser(description='Save embeddings to MegaFace features files')
    parser.add_argument('--model', choices=models_backbones.keys(), type=str, default='rmnet', help='Model type.')
    parser.add_argument('input_dir', help='Path to MegaFace Features')
    parser.add_argument('output_dir', help='Path to FaceScrub Features')
    parser.add_argument('--input_list', default='list.txt', type=str, required=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='CUDA devices to use.')
    parser.add_argument('--snap', type=str, required=True, help='Snapshot to evaluate.')
    parser.add_argument('--noises_list', type=str, default='', required=False, help='A list of the Megaface or Facescrub noises produced by insightface. \
                                                                        See https://github.com/deepinsight/insightface/blob/master/src/megaface/README.md')
    parser.add_argument('--file_ending', help='Ending appended to original photo files. i.e.\
                        11084833664_0.jpg_LBP_100x100.bin => _LBP_100x100.bin', default='_rmnet.bin')
    parser.add_argument('--trillion_format', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_argument(sys.argv[1:]))
