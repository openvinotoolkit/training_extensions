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
import datetime
from functools import partial

import cv2 as cv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as t

from scipy.spatial.distance import cosine
import glog as log
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from nncf.config import Config
from nncf.dynamic_graph import patch_torch_operators
from nncf.algo_selector import create_compression_algorithm

from datasets.lfw import LFW
from utils.utils import load_model_state, get_model_parameters_number, flip_tensor
from utils.augmentation import ResizeNumpy, CenterCropNumpy, NumpyToTensor
from utils.face_align import FivePointsAligner
from model.common import models_backbones


def get_subset(container, subset_bounds):
    """Returns a subset of the given list with respect to the list of bounds"""
    subset = []
    for bound in subset_bounds:
        subset += container[bound[0]: bound[1]]
    return subset


def get_roc(scores_with_gt, n_threshs=400):
    """Computes a ROC cureve on the LFW dataset"""
    thresholds = np.linspace(0., 4., n_threshs)

    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        fp = 0
        tp = 0
        for score_with_gt in scores_with_gt:
            predict_same = score_with_gt['score'] < threshold
            actual_same = score_with_gt['is_same']

            if predict_same and actual_same:
                tp += 1
            elif predict_same and not actual_same:
                fp += 1

        fp_rates.append(float(fp) / len(scores_with_gt) * 2)
        tp_rates.append(float(tp) / len(scores_with_gt) * 2)

    return np.array(fp_rates), np.array(tp_rates)


def get_auc(fprs, tprs):
    """Computes AUC under a ROC curve"""
    sorted_fprs, sorted_tprs = zip(*sorted(zip(*(fprs, tprs))))
    sorted_fprs = list(sorted_fprs)
    sorted_tprs = list(sorted_tprs)
    if sorted_fprs[-1] != 1.0:
        sorted_fprs.append(1.0)
        sorted_tprs.append(sorted_tprs[-1])
    return np.trapz(sorted_tprs, sorted_fprs)


def save_roc(fp_rates, tp_rates, fname):
    assert fp_rates.shape[0] == tp_rates.shape[0]
    with open(fname + '.txt', 'w') as f:
        for i in range(fp_rates.shape[0]):
            f.write('{} {}\n'.format(fp_rates[i], tp_rates[i]))


@torch.no_grad()
def compute_embeddings_lfw(args, dataset, model, batch_size, dump_embeddings=False,
                           pdist=lambda x, y: 1. - F.cosine_similarity(x, y), flipped_embeddings=False):
    """Computes embeddings of all images from the LFW dataset using PyTorch"""
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    scores_with_gt = []
    embeddings = []
    ids = []

    for batch_idx, data in enumerate(tqdm(val_loader, 'Computing embeddings')):
        images_1 = data['img1']
        images_2 = data['img2']
        is_same = data['is_same']
        if torch.cuda.is_available() and args.devices[0] != -1:
            images_1 = images_1.cuda()
            images_2 = images_2.cuda()
        emb_1 = model(images_1)
        emb_2 = model(images_2)
        if flipped_embeddings:
            images_1_flipped = flip_tensor(images_1, 3)
            images_2_flipped = flip_tensor(images_2, 3)
            emb_1_flipped = model(images_1_flipped)
            emb_2_flipped = model(images_2_flipped)
            emb_1 = (emb_1 + emb_1_flipped)*.5
            emb_2 = (emb_2 + emb_2_flipped)*.5
        scores = pdist(emb_1, emb_2).data.cpu().numpy()

        for i, _ in enumerate(scores):
            scores_with_gt.append({'score': scores[i], 'is_same': is_same[i], 'idx': batch_idx*batch_size + i})

        if dump_embeddings:
            id0 = data['id0']
            id1 = data['id1']
            ids.append(id0)
            ids.append(id1)
            to_dump_1 = emb_1.data.cpu()
            to_dump_2 = emb_2.data.cpu()
            embeddings.append(to_dump_1)
            embeddings.append(to_dump_2)

    if dump_embeddings:
        total_emb = np.concatenate(embeddings, axis=0)
        total_ids = np.concatenate(ids, axis=0)
        log_path = './logs/{:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
        writer = SummaryWriter(log_path)
        writer.add_embedding(torch.from_numpy(total_emb), total_ids)

    return scores_with_gt


def compute_embeddings_lfw_ie(args, dataset, model, batch_size=1, dump_embeddings=False,
                              pdist=cosine, flipped_embeddings=False, lm_model=None):
    """Computes embeddings of all images from the LFW dataset using Inference Engine"""
    assert batch_size == 1
    scores_with_gt = []

    for batch_idx, data in enumerate(tqdm(dataset, 'Computing embeddings')):
        images_1 = data['img1']
        images_2 = data['img2']
        if lm_model:
            lm_input_size = tuple(lm_model.get_input_shape()[2:])
            landmarks_1 = lm_model.forward(cv.resize(images_1, lm_input_size)).reshape(-1)
            images_1 = FivePointsAligner.align(images_1, landmarks_1, *images_1.shape[:2], normalize=False, show=False)

            landmarks_2 = lm_model.forward(cv.resize(images_2, lm_input_size)).reshape(-1)
            images_2 = FivePointsAligner.align(images_2, landmarks_2, *images_2.shape[:2], normalize=False)

        is_same = data['is_same']
        emb_1 = model.forward(images_1).reshape(-1)
        emb_2 = model.forward(images_2).reshape(-1)
        score = pdist(emb_1, emb_2)
        scores_with_gt.append({'score': score, 'is_same': is_same, 'idx': batch_idx * batch_size})

    return scores_with_gt


def compute_optimal_thresh(scores_with_gt):
    """Computes an optimal threshold for pairwise face verification"""
    pos_scores = []
    neg_scores = []
    for score_with_gt in scores_with_gt:
        if score_with_gt['is_same']:
            pos_scores.append(score_with_gt['score'])
        else:
            neg_scores.append(score_with_gt['score'])

    hist_pos, bins = np.histogram(np.array(pos_scores), 60)
    hist_neg, _ = np.histogram(np.array(neg_scores), bins)

    intersection_bins = []

    for i in range(1, len(hist_neg)):
        if hist_pos[i - 1] >= hist_neg[i - 1] and 0.05 < hist_pos[i] <= hist_neg[i]:
            intersection_bins.append(bins[i])

    if not intersection_bins:
        intersection_bins.append(0.5)

    return np.mean(intersection_bins)


def evaluate(args, dataset, model, compute_embeddings_fun, val_batch_size=16,
             dump_embeddings=False, roc_fname='', snap_name='', verbose=True, show_failed=False):
    """Computes the LFW score of given model"""
    if verbose and isinstance(model, torch.nn.Module):
        log.info('Face recognition model config:')
        log.info(model)
        log.info('Number of parameters: {}'.format(get_model_parameters_number(model)))

    scores_with_gt = compute_embeddings_fun(args, dataset, model, val_batch_size, dump_embeddings)
    num_pairs = len(scores_with_gt)

    subsets = []
    for i in range(10):
        lower_bnd = i * num_pairs // 10
        upper_bnd = (i + 1) * num_pairs // 10
        subset_test = [(lower_bnd, upper_bnd)]
        subset_train = [(0, lower_bnd), (upper_bnd, num_pairs)]
        subsets.append({'test': subset_test, 'train': subset_train})

    same_scores = []
    diff_scores = []
    val_scores = []
    threshs = []
    mean_fpr = np.zeros(400)
    mean_tpr = np.zeros(400)
    failed_pairs = []

    for subset in tqdm(subsets, '{} evaluation'.format(snap_name), disable=not verbose):
        train_list = get_subset(scores_with_gt, subset['train'])
        optimal_thresh = compute_optimal_thresh(train_list)
        threshs.append(optimal_thresh)

        test_list = get_subset(scores_with_gt, subset['test'])
        same_correct = 0
        diff_correct = 0
        pos_pairs_num = neg_pairs_num = len(test_list) // 2

        for score_with_gt in test_list:
            if score_with_gt['score'] < optimal_thresh and score_with_gt['is_same']:
                same_correct += 1
            elif score_with_gt['score'] >= optimal_thresh and not score_with_gt['is_same']:
                diff_correct += 1

            if score_with_gt['score'] >= optimal_thresh and score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])
            if score_with_gt['score'] < optimal_thresh and not score_with_gt['is_same']:
                failed_pairs.append(score_with_gt['idx'])

        same_scores.append(float(same_correct) / pos_pairs_num)
        diff_scores.append(float(diff_correct) / neg_pairs_num)
        val_scores.append(0.5*(same_scores[-1] + diff_scores[-1]))

        fprs, tprs = get_roc(test_list, mean_fpr.shape[0])
        mean_fpr = mean_fpr + fprs
        mean_tpr = mean_tpr + tprs

    mean_fpr /= 10
    mean_tpr /= 10

    if roc_fname:
        save_roc(mean_tpr, mean_fpr, roc_fname)

    same_acc = np.mean(same_scores)
    diff_acc = np.mean(diff_scores)
    overall_acc = np.mean(val_scores)
    auc = get_auc(mean_fpr, mean_tpr)

    if show_failed:
        log.info('Number of misclassified pairs: {}'.format(len(failed_pairs)))
        for pair in failed_pairs:
            dataset.show_item(pair)

    if verbose:
        log.info('Accuracy/Val_same_accuracy mean: {0:.4f}'.format(same_acc))
        log.info('Accuracy/Val_diff_accuracy mean: {0:.4f}'.format(diff_acc))
        log.info('Accuracy/Val_accuracy mean: {0:.4f}'.format(overall_acc))
        log.info('Accuracy/Val_accuracy std dev: {0:.4f}'.format(np.std(val_scores)))
        log.info('AUC: {0:.4f}'.format(auc))
        log.info('Estimated threshold: {0:.4f}'.format(np.mean(threshs)))
        if args.compr_config and "sparsity_level" in compression_algo.statistics():
            log.info(
                "Sparsity level: {0:.2f}".format(compression_algo.statistics()['sparsity_rate_for_sparsified_modules']))
    return same_acc, diff_acc, overall_acc, auc


def load_test_dataset(arguments):
    """Loads and configures the LFW dataset"""
    input_size = models_backbones[arguments.model]().get_input_res()
    lfw = LFW(arguments.val, arguments.v_list, arguments.v_land)
    assert lfw.use_landmarks
    log.info('Using landmarks for the LFW images.')
    transform = t.Compose([ResizeNumpy(input_size),
                           NumpyToTensor(switch_rb=True)])
    lfw.transform = transform
    return lfw, partial(compute_embeddings_lfw, flipped_embeddings=arguments.flipped_emb)


def main():
    parser = argparse.ArgumentParser(description='Evaluation script for Face Recognition in PyTorch')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='CUDA devices to use.')
    parser.add_argument('--embed_size', type=int, default=128, help='Size of the face embedding.')
    parser.add_argument('--val_data_root', dest='val', required=True, type=str, help='Path to validation data.')
    parser.add_argument('--val_list', dest='v_list', required=True, type=str, help='Path to train data image list.')
    parser.add_argument('--val_landmarks', dest='v_land', default='', required=False, type=str,
                        help='Path to landmarks for the test images.')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size.')
    parser.add_argument('--snap', type=str, required=False, help='Snapshot to evaluate.')
    parser.add_argument('--roc_fname', type=str, default='', help='ROC file.')
    parser.add_argument('--dump_embeddings', action='store_true', help='Dump embeddings to summary writer.')
    parser.add_argument('--dist', choices=['l2', 'cos'], type=str, default='cos', help='Distance.')
    parser.add_argument('--flipped_emb', action='store_true', help='Flipped embedding concatenation trick.')
    parser.add_argument('--show_failed', action='store_true', help='Show misclassified pairs.')
    parser.add_argument('--model', choices=models_backbones.keys(), type=str, default='rmnet', help='Model type.')
    parser.add_argument('--engine', choices=['pt', 'ie'], type=str, default='pt', help='Framework to use for eval.')

    # IE-related options
    parser.add_argument('--fr_model', type=str, required=False)
    parser.add_argument('--lm_model', type=str, required=False)
    parser.add_argument('-pp', '--plugin_dir', type=str, default=None, help='Path to a plugin folder')
    parser.add_argument('-c', '--compr_config', help='Path to a file with compression parameters', required=False)
    args = parser.parse_args()

    if args.engine == 'pt':
        assert args.snap is not None, 'To evaluate PyTorch snapshot, please, specify --snap option.'

        if args.compr_config:
            patch_torch_operators()

        with torch.cuda.device(args.devices[0]):
            data, embeddings_fun = load_test_dataset(args)
            model = models_backbones[args.model](embedding_size=args.embed_size, feature=True)

            if args.compr_config:
                config = Config.from_json(args.compr_config)
                compression_algo = create_compression_algorithm(model, config)
                model = compression_algo.model

            model = load_model_state(model, args.snap, args.devices[0])
            evaluate(args, data, model, embeddings_fun, args.val_batch_size, args.dump_embeddings,
                     args.roc_fname, args.snap, True, args.show_failed)

            if args.compr_config and "sparsity_level" in compression_algo.statistics():
                log.info("Sparsity level: {0:.2f}".format(
                    compression_algo.statistics()['sparsity_rate_for_sparsified_modules']))
    else:
        from utils.ie_tools import load_ie_model

        assert args.fr_model is not None, 'To evaluate IE model, please, specify --fr_model option.'
        fr_model = load_ie_model(args.fr_model, 'CPU', args.plugin_dir)
        lm_model = None
        if args.lm_model:
            lm_model = load_ie_model(args.lm_model, 'CPU', args.plugin_dir)
        input_size = tuple(fr_model.get_input_shape()[2:])

        lfw = LFW(args.val, args.v_list, args.v_land)
        if not lfw.use_landmarks or lm_model:
            lfw.transform = t.Compose([ResizeNumpy(220), CenterCropNumpy(input_size)])
            lfw.use_landmarks = False
        else:
            log.info('Using landmarks for the LFW images.')
            lfw.transform = t.Compose([ResizeNumpy(input_size)])

        evaluate(args, lfw, fr_model, partial(compute_embeddings_lfw_ie, lm_model=lm_model), val_batch_size=1,
                 dump_embeddings=False, roc_fname='', snap_name='', verbose=True, show_failed=False)


if __name__ == '__main__':
    main()
