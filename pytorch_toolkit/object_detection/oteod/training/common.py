# pylint: disable=W1203

import json
import logging
import os
import subprocess
import sys
import tempfile

import torch
import yaml
from mmcv.utils import Config
from oteod import MMDETECTION_TOOLS
from oteod.misc import run_with_termination


def train_internal(config, gpu_num, update_config):
    training_info = {'training_gpu_num': 0}
    if torch.cuda.is_available():
        logging.info('Training on GPUs started ...')
        available_gpu_num = torch.cuda.device_count()
        if available_gpu_num < gpu_num:
            logging.warning(f'available_gpu_num < args.gpu_num: {available_gpu_num} < {gpu_num}')
            logging.warning(f'decreased number of gpu to: {available_gpu_num}')
            gpu_num = available_gpu_num
            sys.stdout.flush()
        run_with_termination(f'{MMDETECTION_TOOLS}/dist_train.sh'
                             f' {config}'
                             f' {gpu_num}'
                             f'{update_config}'.split(' '))
        training_info['training_gpu_num'] = gpu_num
        logging.info('... training on GPUs completed.')
    else:
        logging.info('Training on CPU started ...')
        run_with_termination(f'python {MMDETECTION_TOOLS}/train.py'
                             f' {config}'
                             f'{update_config}'.split(' '))
        logging.info('... training on CPU completed.')

    return training_info


def is_clustering_needed(cfg):
    if cfg.total_epochs > 0:
        return False
    if not hasattr(cfg.model, 'bbox_head') or not cfg.model.bbox_head.type == 'SSDHead':
        return False
    if not cfg.model.bbox_head.anchor_generator.type == 'SSDAnchorGeneratorClustered':
        return False
    return True


def cluster(cfg, config_path, update_config):
    mmdetection_tools = f'{os.path.dirname(__file__)}/../../../../external/mmdetection/tools'
    logging.info('Clustering started...')
    widths = cfg.model.bbox_head.anchor_generator.widths
    n_clust = 0
    for w in widths:
        n_clust += len(w) if isinstance(w, (list, tuple)) else 1
    n_clust = ' --n_clust ' + str(n_clust)

    group_as = ''
    if isinstance(widths[0], (list, tuple)):
        group_as = ' --group_as ' + ' '.join([str(len(w)) for w in widths])

    config = ' --config ' + config_path

    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    out = f' --out {tmp_file.name}'

    if 'pipeline' in cfg.data.train:
        img_shape = [t for t in cfg.data.train.pipeline if t['type'] == 'Resize'][0][
            'img_scale']
    else:
        img_shape = [t for t in cfg.data.train.dataset.pipeline if t['type'] == 'Resize'][0][
            'img_scale']

    img_shape = f' --image_size_wh {img_shape[0]} {img_shape[1]}'

    subprocess.run(f'python {mmdetection_tools}/cluster_boxes.py'
                   f'{config}'
                   f'{n_clust}'
                   f'{group_as}'
                   f'{update_config}'
                   f'{img_shape}'
                   f'{out}'.split(' '), check=True)

    with open(tmp_file.name) as src_file:
        content = json.load(src_file)
        widths, heights = content['widths'], content['heights']

    if not update_config:
        update_config = ' --update_config'
    update_config += f' model.bbox_head.anchor_generator.widths={str(widths).replace(" ", "")}'
    update_config += f' model.bbox_head.anchor_generator.heights={str(heights).replace(" ", "")}'
    logging.info('... clustering completed.')

    return update_config


def train(config, gpu_num, out, update_config):
    """ Main function. """

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Commandline:\n{" ".join(sys.argv)}')

    cfg = Config.fromfile(config)

    update_config = ' '.join([f'{k}={v}' for k, v in update_config.items()])
    update_config = f' --update_config {update_config}' if update_config else ''

    if is_clustering_needed(cfg):
        update_config = cluster(cfg, config, update_config)

    logging.info('Training started ...')
    training_info = train_internal(config, gpu_num, update_config)
    logging.info('... training completed.')

    with open(out, 'a+') as dst_file:
        yaml.dump(training_info, dst_file)
