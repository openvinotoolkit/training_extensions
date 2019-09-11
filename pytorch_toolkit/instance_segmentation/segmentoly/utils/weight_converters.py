"""
 Copyright (c) 2019 Intel Corporation

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
import pickle
from collections import OrderedDict
from shutil import copyfile

import torch

from ..rcnn.backbones.resnet import ResBlock, ResBlockWithFusedBN, ResNetBody


def as_is_converter(src_model_path, dst_model_path):
    copyfile(src_model_path, dst_model_path)


def maskrcnn_benchmark_models_converter(src_model_path, dst_model_path, weights_mapping):
    state_dict = torch.load(src_model_path, map_location='cpu')
    state_dict = state_dict['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in weights_mapping:
            if weights_mapping[k]:
                new_state_dict[weights_mapping[k]] = v
        else:
            logging.warning('Warning! No mapping for {}'.format(k))
    state_dict = new_state_dict
    torch.save(dict(model=state_dict), dst_model_path)


def maskrcnn_benchmark_resnet50_c4_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = MaskRCNNBenchmarkWeightsMapper.resnet((3, 4, 6))
    mapping.update(MaskRCNNBenchmarkWeightsMapper.heads())
    maskrcnn_benchmark_models_converter(src_model_path, dst_model_path, mapping)


def maskrcnn_benchmark_resnet50_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = MaskRCNNBenchmarkWeightsMapper.resnet((3, 4, 6, 3))
    mapping.update(MaskRCNNBenchmarkWeightsMapper.fpn_heads())
    maskrcnn_benchmark_models_converter(src_model_path, dst_model_path, mapping)


def maskrcnn_benchmark_resnet101_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = MaskRCNNBenchmarkWeightsMapper.resnet((3, 4, 23, 3))
    mapping.update(MaskRCNNBenchmarkWeightsMapper.fpn_heads())
    maskrcnn_benchmark_models_converter(src_model_path, dst_model_path, mapping)


class MaskRCNNBenchmarkWeightsMapper(object):
    @staticmethod
    def resnet_stem():
        mapping = {
            'module.backbone.body.stem.conv1.weight': 'backbone.stages.stage_0.conv1.weight',
            'module.backbone.body.stem.bn1.weight': 'backbone.stages.stage_0.bn1.weight',
            'module.backbone.body.stem.bn1.bias': 'backbone.stages.stage_0.bn1.bias',
            'module.backbone.body.stem.bn1.running_mean': 'backbone.stages.stage_0.bn1.running_mean',
            'module.backbone.body.stem.bn1.running_var': 'backbone.stages.stage_0.bn1.running_var'
        }
        return mapping

    @staticmethod
    def resnet_block(stage_idx, block_idx):
        mapping_template = {
            'module.backbone.body.layer{}.{}.conv1.weight': 'backbone.stages.stage_{}.{}.conv1.weight',
            'module.backbone.body.layer{}.{}.bn1.weight': 'backbone.stages.stage_{}.{}.bn1.weight',
            'module.backbone.body.layer{}.{}.bn1.bias': 'backbone.stages.stage_{}.{}.bn1.bias',
            'module.backbone.body.layer{}.{}.bn1.running_mean': 'backbone.stages.stage_{}.{}.bn1.running_mean',
            'module.backbone.body.layer{}.{}.bn1.running_var': 'backbone.stages.stage_{}.{}.bn1.running_var',
            'module.backbone.body.layer{}.{}.conv2.weight': 'backbone.stages.stage_{}.{}.conv2.weight',
            'module.backbone.body.layer{}.{}.bn2.weight': 'backbone.stages.stage_{}.{}.bn2.weight',
            'module.backbone.body.layer{}.{}.bn2.bias': 'backbone.stages.stage_{}.{}.bn2.bias',
            'module.backbone.body.layer{}.{}.bn2.running_mean': 'backbone.stages.stage_{}.{}.bn2.running_mean',
            'module.backbone.body.layer{}.{}.bn2.running_var': 'backbone.stages.stage_{}.{}.bn2.running_var',
            'module.backbone.body.layer{}.{}.conv3.weight': 'backbone.stages.stage_{}.{}.conv3.weight',
            'module.backbone.body.layer{}.{}.bn3.weight': 'backbone.stages.stage_{}.{}.bn3.weight',
            'module.backbone.body.layer{}.{}.bn3.bias': 'backbone.stages.stage_{}.{}.bn3.bias',
            'module.backbone.body.layer{}.{}.bn3.running_mean': 'backbone.stages.stage_{}.{}.bn3.running_mean',
            'module.backbone.body.layer{}.{}.bn3.running_var': 'backbone.stages.stage_{}.{}.bn3.running_var',
        }
        mapping = {k.format(stage_idx, block_idx): v.format(stage_idx, block_idx)
                   for k, v in mapping_template.items()}
        return mapping

    @staticmethod
    def resnet_downscale_block(stage_idx, block_idx):
        mapping_template = {
            'module.backbone.body.layer{}.{}.downsample.0.weight': 'backbone.stages.stage_{}.{}.downsample.0.weight',
            'module.backbone.body.layer{}.{}.downsample.1.weight': 'backbone.stages.stage_{}.{}.downsample.1.weight',
            'module.backbone.body.layer{}.{}.downsample.1.bias': 'backbone.stages.stage_{}.{}.downsample.1.bias',
            'module.backbone.body.layer{}.{}.downsample.1.running_mean': 'backbone.stages.stage_{}.{}.downsample.1.running_mean',
            'module.backbone.body.layer{}.{}.downsample.1.running_var': 'backbone.stages.stage_{}.{}.downsample.1.running_var'
        }
        mapping = {k.format(stage_idx, block_idx): v.format(stage_idx, block_idx)
                   for k, v in mapping_template.items()}
        mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_block(stage_idx, block_idx))
        return mapping

    @staticmethod
    def resnet_stage(stage_idx, blocks_num):
        mapping = {}
        mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_downscale_block(stage_idx, 0))
        for i in range(1, blocks_num):
            mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_block(stage_idx, i))
        return mapping

    @staticmethod
    def resnet(blocks_per_stage):
        mapping = {}
        mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_stem())
        for stage_idx, blocks_num in enumerate(blocks_per_stage, 1):
            mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_stage(stage_idx, blocks_num))
        return mapping

    @staticmethod
    def fpn_heads():
        mapping = {
            'module.backbone.fpn.fpn_inner1.weight': 'fpn.topdown_lateral.2.conv_lateral.weight',
            'module.backbone.fpn.fpn_inner1.bias': 'fpn.topdown_lateral.2.conv_lateral.bias',
            'module.backbone.fpn.fpn_layer1.weight': 'fpn.posthoc.3.weight',
            'module.backbone.fpn.fpn_layer1.bias': 'fpn.posthoc.3.bias',
            'module.backbone.fpn.fpn_inner2.weight': 'fpn.topdown_lateral.1.conv_lateral.weight',
            'module.backbone.fpn.fpn_inner2.bias': 'fpn.topdown_lateral.1.conv_lateral.bias',
            'module.backbone.fpn.fpn_layer2.weight': 'fpn.posthoc.2.weight',
            'module.backbone.fpn.fpn_layer2.bias': 'fpn.posthoc.2.bias',
            'module.backbone.fpn.fpn_inner3.weight': 'fpn.topdown_lateral.0.conv_lateral.weight',
            'module.backbone.fpn.fpn_inner3.bias': 'fpn.topdown_lateral.0.conv_lateral.bias',
            'module.backbone.fpn.fpn_layer3.weight': 'fpn.posthoc.1.weight',
            'module.backbone.fpn.fpn_layer3.bias': 'fpn.posthoc.1.bias',
            'module.backbone.fpn.fpn_inner4.weight': 'fpn.conv_top.weight',
            'module.backbone.fpn.fpn_inner4.bias': 'fpn.conv_top.bias',
            'module.backbone.fpn.fpn_layer4.weight': 'fpn.posthoc.0.weight',
            'module.backbone.fpn.fpn_layer4.bias': 'fpn.posthoc.0.bias',
            'module.rpn.anchor_generator.cell_anchors.0': '',
            'module.rpn.anchor_generator.cell_anchors.1': '',
            'module.rpn.anchor_generator.cell_anchors.2': '',
            'module.rpn.anchor_generator.cell_anchors.3': '',
            'module.rpn.anchor_generator.cell_anchors.4': '',
            'module.rpn.head.conv.weight': 'rpn.conv.weight',
            'module.rpn.head.conv.bias': 'rpn.conv.bias',
            'module.rpn.head.cls_logits.weight': 'rpn.cls_score.weight',
            'module.rpn.head.cls_logits.bias': 'rpn.cls_score.bias',
            'module.rpn.head.bbox_pred.weight': 'rpn.bbox_deltas.weight',
            'module.rpn.head.bbox_pred.bias': 'rpn.bbox_deltas.bias',
            'module.roi_heads.box.feature_extractor.fc6.weight': 'detection_head.fc1.weight',
            'module.roi_heads.box.feature_extractor.fc6.bias': 'detection_head.fc1.bias',
            'module.roi_heads.box.feature_extractor.fc7.weight': 'detection_head.fc2.weight',
            'module.roi_heads.box.feature_extractor.fc7.bias': 'detection_head.fc2.bias',
            'module.roi_heads.box.predictor.cls_score.weight': 'detection_head.cls_score.weight',
            'module.roi_heads.box.predictor.cls_score.bias': 'detection_head.cls_score.bias',
            'module.roi_heads.box.predictor.bbox_pred.weight': 'detection_head.bbox_pred.weight',
            'module.roi_heads.box.predictor.bbox_pred.bias': 'detection_head.bbox_pred.bias',
            'module.roi_heads.mask.feature_extractor.mask_fcn1.weight': 'mask_head.conv_fcn.0.weight',
            'module.roi_heads.mask.feature_extractor.mask_fcn1.bias': 'mask_head.conv_fcn.0.bias',
            'module.roi_heads.mask.feature_extractor.mask_fcn2.weight': 'mask_head.conv_fcn.2.weight',
            'module.roi_heads.mask.feature_extractor.mask_fcn2.bias': 'mask_head.conv_fcn.2.bias',
            'module.roi_heads.mask.feature_extractor.mask_fcn3.weight': 'mask_head.conv_fcn.4.weight',
            'module.roi_heads.mask.feature_extractor.mask_fcn3.bias': 'mask_head.conv_fcn.4.bias',
            'module.roi_heads.mask.feature_extractor.mask_fcn4.weight': 'mask_head.conv_fcn.6.weight',
            'module.roi_heads.mask.feature_extractor.mask_fcn4.bias': 'mask_head.conv_fcn.6.bias',
            'module.roi_heads.mask.predictor.conv5_mask.weight': 'mask_head.upconv.weight',
            'module.roi_heads.mask.predictor.conv5_mask.bias': 'mask_head.upconv.bias',
            'module.roi_heads.mask.predictor.mask_fcn_logits.weight': 'mask_head.segm.weight',
            'module.roi_heads.mask.predictor.mask_fcn_logits.bias': 'mask_head.segm.bias'
        }
        return mapping

    @staticmethod
    def heads():
        mapping = {
            'module.rpn.anchor_generator.cell_anchors.0': '',
            'module.rpn.head.conv.weight': 'rpn.conv.weight',
            'module.rpn.head.conv.bias': 'rpn.conv.bias',
            'module.rpn.head.cls_logits.weight': 'rpn.cls_score.weight',
            'module.rpn.head.cls_logits.bias': 'rpn.cls_score.bias',
            'module.rpn.head.bbox_pred.weight': 'rpn.bbox_deltas.weight',
            'module.rpn.head.bbox_pred.bias': 'rpn.bbox_deltas.bias',

            'module.roi_heads.box.predictor.cls_score.weight': 'detection_head.cls_score.weight',
            'module.roi_heads.box.predictor.cls_score.bias': 'detection_head.cls_score.bias',
            'module.roi_heads.box.predictor.bbox_pred.weight': 'detection_head.bbox_pred.weight',
            'module.roi_heads.box.predictor.bbox_pred.bias': 'detection_head.bbox_pred.bias',

            'module.roi_heads.mask.predictor.conv5_mask.weight': 'mask_head.upconv5.weight',
            'module.roi_heads.mask.predictor.conv5_mask.bias': 'mask_head.upconv5.bias',
            'module.roi_heads.mask.predictor.mask_fcn_logits.weight': 'mask_head.segm.weight',
            'module.roi_heads.mask.predictor.mask_fcn_logits.bias': 'mask_head.segm.bias'
        }
        common_feature_extractor_mapping = MaskRCNNBenchmarkWeightsMapper.resnet_downscale_block(4, 0)
        common_feature_extractor_mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_block(4, 1))
        common_feature_extractor_mapping.update(MaskRCNNBenchmarkWeightsMapper.resnet_block(4, 2))
        mapping.update({k.replace('backbone.body', 'roi_heads.box.feature_extractor.head'):
                            v.replace('backbone.stages.stage_4', 'common_detection_mask_head')
                        for k, v in common_feature_extractor_mapping.items()})
        mapping.update({k.replace('backbone.body', 'roi_heads.mask.feature_extractor.head'): ''
                        for k, v in common_feature_extractor_mapping.items()})
        return mapping


def detectron_models_converter(src_model_path, dst_model_path, weights_mapping):
    with open(src_model_path, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')

    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']
    state_dict = OrderedDict()
    for k, v in src_blobs.items():
        if k in weights_mapping:
            if weights_mapping[k]:
                state_dict[weights_mapping[k]] = torch.as_tensor(v)
        elif k.endswith('_momentum'):
            pass
        else:
            logging.warning('Warning! No mapping for {}'.format(k))
    torch.save(dict(model=state_dict), dst_model_path)


def detectron_resnet50_c4_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 6), prefix='backbone.')
    mapping.update(DetectronWeightsMapper.heads())
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet50_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 6, 3), prefix='backbone.')
    mapping.update(DetectronWeightsMapper.fpn_heads((3, 4, 6, 3)))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet50_gn_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 6, 3), prefix='backbone.', normalization='gn')
    mapping.update(DetectronWeightsMapper.fpn_heads((3, 4, 6, 3)))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet101_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 23, 3), prefix='backbone.')
    mapping.update(DetectronWeightsMapper.fpn_heads((3, 4, 23, 3)))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet101_gn_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 23, 3), prefix='backbone.', normalization='gn')
    mapping.update(DetectronWeightsMapper.fpn_heads((3, 4, 23, 3)))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet152_fpn_mask_rcnn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 8, 36, 3), prefix='backbone.')
    mapping.update(DetectronWeightsMapper.fpn_heads((3, 8, 36, 3)))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet50_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 6, 3))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet50_gn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 6, 3), normalization='gn')
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet101_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 23, 3))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet101_gn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 4, 23, 3), normalization='gn')
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet152_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 8, 36, 3))
    detectron_models_converter(src_model_path, dst_model_path, mapping)


def detectron_resnet152_gn_converter(src_model_path, dst_model_path):
    mapping = DetectronWeightsMapper.resnet((3, 8, 36, 3), normalization='gn')
    detectron_models_converter(src_model_path, dst_model_path, mapping)


class DetectronWeightsMapper(object):
    @staticmethod
    def resnet_stem(prefix='', normalization='bn'):
        mapping_template_weights = {
            'conv1_w': '{}stages.stage_0.conv1.weight',
            'conv1_b': '',
        }
        mapping = {k: v.format(prefix)
                   for k, v in mapping_template_weights.items()}
        p = 'res_' if normalization == 'bn' else ''
        mapping_template_norm = {
            '{}conv1_{}_s': '{}stages.stage_0.bn1.weight',
            '{}conv1_{}_b': '{}stages.stage_0.bn1.bias',
        }
        mapping.update({k.format(p, normalization): v.format(prefix)
                        for k, v in mapping_template_norm.items()})
        return mapping

    @staticmethod
    def resnet_block(stage_idx, block_idx, prefix='', normalization='bn'):
        mapping_template_weights = {
            'res{}_{}_branch2a_w': '{}stages.stage_{}.{}.conv1.weight',
            'res{}_{}_branch2a_b': '',
            'res{}_{}_branch2b_w': '{}stages.stage_{}.{}.conv2.weight',
            'res{}_{}_branch2b_b': '',
            'res{}_{}_branch2c_w': '{}stages.stage_{}.{}.conv3.weight',
            'res{}_{}_branch2c_b': '',
        }
        mapping = {k.format(stage_idx + 1, block_idx): v.format(prefix, stage_idx, block_idx)
                   for k, v in mapping_template_weights.items()}
        mapping_template_norm = {
            'res{}_{}_branch2a_{}_s': '{}stages.stage_{}.{}.bn1.weight',
            'res{}_{}_branch2a_{}_b': '{}stages.stage_{}.{}.bn1.bias',
            'res{}_{}_branch2b_{}_s': '{}stages.stage_{}.{}.bn2.weight',
            'res{}_{}_branch2b_{}_b': '{}stages.stage_{}.{}.bn2.bias',
            'res{}_{}_branch2c_{}_s': '{}stages.stage_{}.{}.bn3.weight',
            'res{}_{}_branch2c_{}_b': '{}stages.stage_{}.{}.bn3.bias',
        }
        mapping.update({k.format(stage_idx + 1, block_idx, normalization):
                            v.format(prefix, stage_idx, block_idx)
                        for k, v in mapping_template_norm.items()})
        return mapping

    @staticmethod
    def resnet_downscale_block(stage_idx, block_idx, prefix='', normalization='bn'):
        mapping_template_weights = {
            'res{}_{}_branch1_w': '{}stages.stage_{}.{}.downsample.0.weight',
            'res{}_{}_branch1_b': '',
        }
        mapping = {k.format(stage_idx + 1, block_idx): v.format(prefix, stage_idx, block_idx)
                   for k, v in mapping_template_weights.items()}
        mapping_template_norm = {
            'res{}_{}_branch1_{}_s': '{}stages.stage_{}.{}.downsample.1.weight',
            'res{}_{}_branch1_{}_b': '{}stages.stage_{}.{}.downsample.1.bias',
        }
        mapping.update({k.format(stage_idx + 1, block_idx, normalization): v.format(prefix, stage_idx, block_idx)
                        for k, v in mapping_template_norm.items()})
        mapping.update(DetectronWeightsMapper.resnet_block(stage_idx, block_idx, prefix, normalization))
        return mapping

    @staticmethod
    def resnet_stage(stage_idx, blocks_num, prefix='', normalization='bn'):
        mapping = DetectronWeightsMapper.resnet_downscale_block(stage_idx, 0, prefix, normalization)
        for i in range(1, blocks_num):
            mapping.update(DetectronWeightsMapper.resnet_block(stage_idx, i, prefix, normalization))
        return mapping

    @staticmethod
    def resnet(blocks_per_stage, prefix='', normalization='bn'):
        mapping = DetectronWeightsMapper.resnet_stem(prefix, normalization)
        for stage_idx, blocks_num in enumerate(blocks_per_stage, 1):
            mapping.update(DetectronWeightsMapper.resnet_stage(stage_idx, blocks_num, prefix, normalization))
        return mapping

    @staticmethod
    def fpn_heads(blocks_per_stage):
        assert len(blocks_per_stage) >= 4
        mapping = {
            'fpn_inner_res5_{}_sum_w'.format(blocks_per_stage[3] - 1): 'fpn.conv_top.weight',
            'fpn_inner_res5_{}_sum_b'.format(blocks_per_stage[3] - 1): 'fpn.conv_top.bias',
            'fpn_inner_res4_{}_sum_lateral_w'.format(blocks_per_stage[2] - 1): 'fpn.topdown_lateral.0.conv_lateral.weight',
            'fpn_inner_res4_{}_sum_lateral_b'.format(blocks_per_stage[2] - 1): 'fpn.topdown_lateral.0.conv_lateral.bias',
            'fpn_inner_res3_{}_sum_lateral_w'.format(blocks_per_stage[1] - 1): 'fpn.topdown_lateral.1.conv_lateral.weight',
            'fpn_inner_res3_{}_sum_lateral_b'.format(blocks_per_stage[1] - 1): 'fpn.topdown_lateral.1.conv_lateral.bias',
            'fpn_inner_res2_{}_sum_lateral_w'.format(blocks_per_stage[0] - 1): 'fpn.topdown_lateral.2.conv_lateral.weight',
            'fpn_inner_res2_{}_sum_lateral_b'.format(blocks_per_stage[0] - 1): 'fpn.topdown_lateral.2.conv_lateral.bias',
            'fpn_res5_{}_sum_w'.format(blocks_per_stage[3] - 1): 'fpn.posthoc.0.weight',
            'fpn_res5_{}_sum_b'.format(blocks_per_stage[3] - 1): 'fpn.posthoc.0.bias',
            'fpn_res4_{}_sum_w'.format(blocks_per_stage[2] - 1): 'fpn.posthoc.1.weight',
            'fpn_res4_{}_sum_b'.format(blocks_per_stage[2] - 1): 'fpn.posthoc.1.bias',
            'fpn_res3_{}_sum_w'.format(blocks_per_stage[1] - 1): 'fpn.posthoc.2.weight',
            'fpn_res3_{}_sum_b'.format(blocks_per_stage[1] - 1): 'fpn.posthoc.2.bias',
            'fpn_res2_{}_sum_w'.format(blocks_per_stage[0] - 1): 'fpn.posthoc.3.weight',
            'fpn_res2_{}_sum_b'.format(blocks_per_stage[0] - 1): 'fpn.posthoc.3.bias',

            'conv_rpn_fpn2_w': 'rpn.conv.weight',
            'conv_rpn_fpn2_b': 'rpn.conv.bias',
            'rpn_cls_logits_fpn2_w': 'rpn.cls_score.weight',
            'rpn_cls_logits_fpn2_b': 'rpn.cls_score.bias',
            'rpn_bbox_pred_fpn2_w': 'rpn.bbox_deltas.weight',
            'rpn_bbox_pred_fpn2_b': 'rpn.bbox_deltas.bias',

            'fc6_w': 'detection_head.fc1.weight',
            'fc6_b': 'detection_head.fc1.bias',
            'fc7_w': 'detection_head.fc2.weight',
            'fc7_b': 'detection_head.fc2.bias',
            'cls_score_w': 'detection_head.cls_score.weight',
            'cls_score_b': 'detection_head.cls_score.bias',
            'bbox_pred_w': 'detection_head.bbox_pred.weight',
            'bbox_pred_b': 'detection_head.bbox_pred.bias',

            '_[mask]_fcn1_w': 'mask_head.conv_fcn.0.weight',
            '_[mask]_fcn1_b': 'mask_head.conv_fcn.0.bias',
            '_[mask]_fcn2_w': 'mask_head.conv_fcn.2.weight',
            '_[mask]_fcn2_b': 'mask_head.conv_fcn.2.bias',
            '_[mask]_fcn3_w': 'mask_head.conv_fcn.4.weight',
            '_[mask]_fcn3_b': 'mask_head.conv_fcn.4.bias',
            '_[mask]_fcn4_w': 'mask_head.conv_fcn.6.weight',
            '_[mask]_fcn4_b': 'mask_head.conv_fcn.6.bias',
            'conv5_mask_w': 'mask_head.upconv.weight',
            'conv5_mask_b': 'mask_head.upconv.bias',
            'mask_fcn_logits_w': 'mask_head.segm.weight',
            'mask_fcn_logits_b': 'mask_head.segm.bias'
        }
        return mapping

    @staticmethod
    def heads():
        mapping = {
            'conv_rpn_w': 'rpn.conv.weight',
            'conv_rpn_b': 'rpn.conv.bias',
            'rpn_cls_logits_w': 'rpn.cls_score.weight',
            'rpn_cls_logits_b': 'rpn.cls_score.bias',
            'rpn_bbox_pred_w': 'rpn.bbox_deltas.weight',
            'rpn_bbox_pred_b': 'rpn.bbox_deltas.bias',

            'res5_0_branch2a_w': 'common_detection_mask_head.0.conv1.weight',
            'res5_0_branch2a_b': '',
            'res5_0_branch2a_bn_s': 'common_detection_mask_head.0.bn1.weight',
            'res5_0_branch2a_bn_b': 'common_detection_mask_head.0.bn1.bias',
            'res5_0_branch2b_w': 'common_detection_mask_head.0.conv2.weight',
            'res5_0_branch2b_b': '',
            'res5_0_branch2b_bn_s': 'common_detection_mask_head.0.bn2.weight',
            'res5_0_branch2b_bn_b': 'common_detection_mask_head.0.bn2.bias',
            'res5_0_branch2c_w': 'common_detection_mask_head.0.conv3.weight',
            'res5_0_branch2c_b': '',
            'res5_0_branch2c_bn_s': 'common_detection_mask_head.0.bn3.weight',
            'res5_0_branch2c_bn_b': 'common_detection_mask_head.0.bn3.bias',
            'res5_0_branch1_w': 'common_detection_mask_head.0.downsample.0.weight',
            'res5_0_branch1_b': '',
            'res5_0_branch1_bn_s': 'common_detection_mask_head.0.downsample.1.weight',
            'res5_0_branch1_bn_b': 'common_detection_mask_head.0.downsample.1.bias',
            'res5_1_branch2a_w': 'common_detection_mask_head.1.conv1.weight',
            'res5_1_branch2a_b': '',
            'res5_1_branch2a_bn_s': 'common_detection_mask_head.1.bn1.weight',
            'res5_1_branch2a_bn_b': 'common_detection_mask_head.1.bn1.bias',
            'res5_1_branch2b_w': 'common_detection_mask_head.1.conv2.weight',
            'res5_1_branch2b_b': '',
            'res5_1_branch2b_bn_s': 'common_detection_mask_head.1.bn2.weight',
            'res5_1_branch2b_bn_b': 'common_detection_mask_head.1.bn2.bias',
            'res5_1_branch2c_w': 'common_detection_mask_head.1.conv3.weight',
            'res5_1_branch2c_b': '',
            'res5_1_branch2c_bn_s': 'common_detection_mask_head.1.bn3.weight',
            'res5_1_branch2c_bn_b': 'common_detection_mask_head.1.bn3.bias',
            'res5_2_branch2a_w': 'common_detection_mask_head.2.conv1.weight',
            'res5_2_branch2a_b': '',
            'res5_2_branch2a_bn_s': 'common_detection_mask_head.2.bn1.weight',
            'res5_2_branch2a_bn_b': 'common_detection_mask_head.2.bn1.bias',
            'res5_2_branch2b_w': 'common_detection_mask_head.2.conv2.weight',
            'res5_2_branch2b_b': '',
            'res5_2_branch2b_bn_s': 'common_detection_mask_head.2.bn2.weight',
            'res5_2_branch2b_bn_b': 'common_detection_mask_head.2.bn2.bias',
            'res5_2_branch2c_w': 'common_detection_mask_head.2.conv3.weight',
            'res5_2_branch2c_b': '',
            'res5_2_branch2c_bn_s': 'common_detection_mask_head.2.bn3.weight',
            'res5_2_branch2c_bn_b': 'common_detection_mask_head.2.bn3.bias',

            'cls_score_w': 'detection_head.cls_score.weight',
            'cls_score_b': 'detection_head.cls_score.bias',
            'bbox_pred_w': 'detection_head.bbox_pred.weight',
            'bbox_pred_b': 'detection_head.bbox_pred.bias',

            'conv5_mask_w': 'mask_head.upconv5.weight',
            'conv5_mask_b': 'mask_head.upconv5.bias',
            'mask_fcn_logits_w': 'mask_head.segm.weight',
            'mask_fcn_logits_b': 'mask_head.segm.bias'
        }
        return mapping


def fuse_conv_and_bn(conv, bn, conv_bn_fused=None):
    if conv_bn_fused is None:
        conv_bn_fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True
        )
    stddev_bn = torch.sqrt(bn.eps + bn.running_var)
    w_bn = bn.weight.div(stddev_bn)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(stddev_bn)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))

    conv_bn_fused.weight.copy_(torch.mm(torch.diag(w_bn), w_conv).view(conv_bn_fused.weight.size()))
    conv_bn_fused.bias.copy_(b_conv.mul(w_bn) + b_bn)
    return conv_bn_fused


def fuse_bns_in_resblock(src_block, dst_block):
    assert isinstance(src_block, ResBlock)
    assert isinstance(dst_block, ResBlockWithFusedBN)

    with torch.no_grad():
        # Main branch.
        fuse_conv_and_bn(src_block.conv1, src_block.bn1, dst_block.conv1)
        fuse_conv_and_bn(src_block.conv2, src_block.bn2, dst_block.conv2)
        fuse_conv_and_bn(src_block.conv3, src_block.bn3, dst_block.conv3)

        # Residual branch.
        if src_block.downsample is not None:
            assert hasattr(dst_block, 'downsample')
            fuse_conv_and_bn(src_block.downsample[0], src_block.downsample[1], dst_block.downsample)

    return dst_block


def fuse_bns_resnet(resnet):
    assert isinstance(resnet, ResNetBody)
    resnet_fused_bn = ResNetBody(block_counts=resnet.block_counts, res_block=ResBlockWithFusedBN,
                                 num_groups=resnet.num_groups, width_per_group=resnet.width_per_group,
                                 res5_dilation=resnet.res5_dilation)
    resnet_fused_bn.load_state_dict(resnet.state_dict(), strict=False)
    dst_modules = dict(resnet_fused_bn.named_modules())
    for name, m in resnet.named_modules():
        if isinstance(m, ResBlock):
            assert name in dst_modules, name
            fuse_bns_in_resblock(m, dst_modules[name])
    return resnet_fused_bn
