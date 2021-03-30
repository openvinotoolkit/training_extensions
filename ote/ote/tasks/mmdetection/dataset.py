from copy import deepcopy
import os.path as osp

import numpy as np
import mmcv
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.pipelines import Compose
from pycocotools.coco import COCO

from ote.interfaces.dataset import IDataset


class CocoDataSource(IDataset):

    def __init__(self, ann_file, data_root):
        self.coco = COCO(ann_file)
        self.data_root = data_root

        self.filenames = []

        for i in sorted(self.coco.imgs.keys()):
            info = self.coco.load_imgs([i])[0]
            self.filenames.append(info['file_name'])

        self.to_float32 = True
        self.color_type = 'color'
        self.file_client_args=dict(backend='disk')
        self.file_client = None

    def __load_image(self, ori_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if self.data_root is not None:
            filename = osp.join(self.data_root, ori_filename)
        else:
            filename = ori_filename

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results = {}

        results['filename'] = filename
        results['ori_filename'] = ori_filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __getitem__(self, indx) -> dict:
        return self.__load_image(self.filenames[indx])

    def __len__(self) -> int:
        return len(self.coco.imgs)

    def get_annotation(self) -> list:
        return self.coco

@DATASETS.register_module()
class CocoDataset2(CocoDataset):

    def __init__(self,                  # pylint: disable=super-init-not-called
                 data_source,
                 pipeline,
                 classes=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 min_size=None):
        self.data_source = data_source
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.img_prefix = None
        self.seg_prefix = None
        self.proposal_file = None

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.data_source.get_annotation())
        # filter data infos if classes are customized
        if self.custom_classes:
            self.data_infos = self.get_subset_by_classes()

        self.proposals = None
        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline

        self.pipeline = Compose([transform for transform in pipeline if transform.type != 'LoadImageFromFile'])
        self.min_size = min_size

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def get_image_from_source(self, img_info):
        return self.data_source[self.coco_id_to_data_source_id[img_info['id']]]

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        result_from_source = self.get_image_from_source(img_info)
        assert result_from_source['ori_filename'] == img_info['filename'], \
            f"{result_from_source['ori_filename']} vs {img_info['filename']}"
        results.update(result_from_source)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        result_from_source = self.get_image_from_source(img_info)
        assert result_from_source['ori_filename'] == img_info['filename'], \
            f"{result_from_source['ori_filename']} vs {img_info['filename']}"
        results.update(result_from_source)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def load_annotations(self, coco_ann):
        self.coco = deepcopy(coco_ann)
        self.coco_id_to_data_source_id = {k:i for i, k in enumerate(sorted(self.coco.imgs.keys()))}
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos
