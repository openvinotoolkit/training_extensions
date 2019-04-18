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

import torch

from .roi_align import roi_align
from .roi_distributor import redistribute_rois


def topk_rois_logic(image_rois, image_rois_probs, max_rois):
    topk_indices = torch.topk(image_rois_probs, min(max_rois, image_rois_probs.shape[0]))[1]
    image_rois = image_rois[topk_indices]
    return image_rois


class TopKROIsStub(torch.autograd.Function):
    @staticmethod
    def symbolic(g, image_rois, image_rois_probs, max_rois):
        return g.op('ExperimentalDetectronTopKROIs', image_rois, image_rois_probs, max_rois_i=max_rois)

    @staticmethod
    def forward(ctx, image_rois, image_rois_probs, max_rois):
        return topk_rois_logic(image_rois, image_rois_probs, max_rois)


def topk_rois(image_rois, image_rois_probs, max_rois, use_stub=False):
    func = TopKROIsStub.apply if use_stub else topk_rois_logic
    return func(image_rois, image_rois_probs, max_rois)


def extract_roi_features(rois, feature_pyramid, pyramid_scales, output_size=7, sampling_ratio=2,
                         distribute_rois_between_levels=True, preserve_rois_order=True, use_stub=False):
    func = extract_roi_features_single_image_stub if use_stub else extract_roi_features_single_image
    roi_features = []
    for i, image_rois in enumerate(rois):
        if len(image_rois):
            image_roi_features, image_rois = func(image_rois, feature_pyramid, pyramid_scales,
                                                  image_id=i, output_size=output_size, sampling_ratio=sampling_ratio,
                                                  distribute_rois_between_levels=distribute_rois_between_levels,
                                                  preserve_rois_order=preserve_rois_order)
        else:
            image_roi_features = None
        roi_features.append(image_roi_features)
        rois[i] = image_rois

    return roi_features, rois


def extract_roi_features_single_image_stub(image_rois, feature_pyramid, pyramid_scales,
                                           image_id, output_size, sampling_ratio,
                                           distribute_rois_between_levels=True, preserve_rois_order=True):
    class ROIFeatureExtractorStub(torch.autograd.Function):
        @staticmethod
        def symbolic(g, rois, *features):
            return g.op('ExperimentalDetectronROIFeatureExtractor', rois, *features,
                        pyramid_scales_i=pyramid_scales, image_id_i=image_id,
                        output_size_i=output_size, sampling_ratio_i=sampling_ratio,
                        distribute_rois_between_levels_i=int(distribute_rois_between_levels),
                        preserve_rois_order_i=int(preserve_rois_order), outputs=2)

        @staticmethod
        def forward(ctx, rois, *features):
            return extract_roi_features_single_image(rois, features,
                                                     pyramid_scales, image_id, output_size,
                                                     sampling_ratio, distribute_rois_between_levels,
                                                     preserve_rois_order)

    return ROIFeatureExtractorStub.apply(image_rois, *feature_pyramid)


def extract_roi_features_single_image(image_rois, feature_pyramid, pyramid_scales, image_id,
                                      output_size, sampling_ratio, distribute_rois_between_levels=True,
                                      preserve_rois_order=True):
    with torch.no_grad():
        if distribute_rois_between_levels:
            # Split proposals between feature pyramid levels.
            rois_to_levels = redistribute_rois(image_rois, levels_num=len(feature_pyramid), canonical_level=2)
            # Permute rois so that rois for each level are stored consecutively.
            all_rois_to_levels = torch.cat(rois_to_levels, dim=0)
            image_rois_reordered = image_rois[all_rois_to_levels]
            # Split Tensor of reordered ROIs to a list of Tensor.
            # Basically do the following
            # rois_per_level = image_rois_reordered.split([len(rois_ids) for rois_ids in rois_to_levels], dim=0)
            # but as long as Tensor.split handles empty slices incorrectly, do it in a more verbose way.
            rois_per_level = []
            i = 0
            for j in [len(rois_ids) for rois_ids in rois_to_levels]:
                rois_per_level.append(image_rois_reordered[i:i + j])
                i += j
            assert i == image_rois.shape[0]
        else:
            image_rois_reordered = image_rois
            rois_per_level = [image_rois for _ in feature_pyramid]

    # Extract features for ROIs from corresponding feature pyramid levels.
    image_roi_features = []
    for rois, feature_map, scale in zip(rois_per_level, feature_pyramid, pyramid_scales):
        if rois.numel():
            fmap = feature_map[image_id:image_id + 1]
            image_roi_features.append(roi_align(fmap, [rois, ],
                                                aligned_height=output_size, aligned_width=output_size,
                                                spatial_scale=float(1 / scale),
                                                sampling_ratio=sampling_ratio))
    if distribute_rois_between_levels:
        image_roi_features = torch.cat(image_roi_features, dim=0)
        if preserve_rois_order:
            # Return back to original rois order.
            reverse_indices = torch.sort(all_rois_to_levels)[1]
            image_roi_features = image_roi_features.index_select(0, reverse_indices)
    else:
        image_roi_features = torch.stack(image_roi_features, dim=0)
        image_rois = image_rois_reordered

    return image_roi_features, image_rois
