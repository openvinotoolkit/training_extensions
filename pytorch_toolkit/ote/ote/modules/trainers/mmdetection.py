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

import logging
import subprocess
import tempfile

from ote import MMDETECTION_TOOLS

from .base import BaseTrainer
from ..registry import TRAINERS


@TRAINERS.register_module()
class MMDetectionTrainer(BaseTrainer):
    def __init__(self):
        super(MMDetectionTrainer, self).__init__()

    def _get_tools_dir(self):
        return MMDETECTION_TOOLS

    def _add_extra_args(self, cfg, config_path, update_config):
        if self.__is_clustering_needed(cfg):
            update_config = self.__cluster(cfg, config, update_config)

        return update_config

    @staticmethod
    def __is_clustering_needed(cfg):
        if cfg.total_epochs > 0:
            return False
        if not hasattr(cfg.model, 'bbox_head') or not cfg.model.bbox_head.type == 'SSDHead':
            return False
        if not cfg.model.bbox_head.anchor_generator.type == 'SSDAnchorGeneratorClustered':
            return False
        return True

    @staticmethod
    def __cluster(cfg, config_path, update_config):
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

        subprocess.run(f'python {MMDETECTION_TOOLS}/cluster_boxes.py'
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
