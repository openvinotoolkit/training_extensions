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

from .nms_function import nms
from ..utils.boxes import bbox_transform, clip_boxes_to_image


def generate_proposals(all_anchors, rpn_output, im_info,
                       pre_nms_count=12000, post_nms_count=2000, nms_threshold=0.7, min_size=0,
                       force_max_output_size=False):
    bbox_deltas, _, scores = rpn_output
    device_id = scores.device
    assert device_id == bbox_deltas.device

    batch_size = scores.shape[0]
    assert batch_size == bbox_deltas.shape[0]

    if not (all_anchors.dim() == 2 and all_anchors.shape[1] == 4):
        all_anchors = all_anchors.view(-1, 4)

    rois = []
    roi_probs = []
    for idx in range(batch_size):
        im_boxes, im_probs = GenerateProposalsSingleImage.apply(im_info[idx], all_anchors,
                                                                bbox_deltas[idx], scores[idx],
                                                                pre_nms_count, post_nms_count,
                                                                nms_threshold, min_size, force_max_output_size)
        rois.append(im_boxes)
        roi_probs.append(im_probs)

    return rois, roi_probs


def generate_proposals_single_image(im_info, all_anchors, bbox_deltas, scores,
                                    pre_nms_count=12000, post_nms_count=2000, nms_threshold=0.7, min_size=0,
                                    force_max_output_size=False):
    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #   - bbox deltas will be (4 * A, H, W) format from conv output
    #   - transpose to (H, W, 4 * A)
    #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
    #     in slowest to fastest order to match the enumerated anchors
    bbox_deltas = bbox_deltas.permute(1, 2, 0).contiguous().view(-1, 4)

    # Same story for the scores:
    #   - scores are (A, H, W) format from conv output
    #   - transpose to (H, W, A)
    #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
    #     to match the order of anchors and bbox_deltas
    scores = scores.permute(1, 2, 0).contiguous().view(-1)

    assert not torch.isnan(scores).any(), 'Scores blob contains NaN values.'
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top self._pre_nms_count (e.g. 6000)
    if pre_nms_count <= 0 or pre_nms_count >= scores.numel():
        sorted_scores, order = torch.sort(scores, dim=0, descending=True)
    else:
        # Avoid sorting possibly large arrays; First partition to get top K
        # unsorted and then sort just those (~20x faster for 200k scores)
        sorted_scores, order = torch.topk(scores, pre_nms_count, largest=True, sorted=True)
    bbox_deltas = torch.index_select(bbox_deltas, 0, order)
    all_anchors = torch.index_select(all_anchors, 0, order)
    scores = sorted_scores

    # Transform anchors into proposals via bbox transformations
    proposals = bbox_transform(all_anchors, bbox_deltas, weights=None)

    # 2. clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    proposals = clip_boxes_to_image(proposals, int(im_info[0]), int(im_info[1]))

    # 3. remove predicted boxes with either height or width < min_size
    keep = filter_boxes(proposals, min_size, im_info)
    torch.index_select(proposals, 0, keep, out=proposals)
    torch.index_select(scores, 0, keep, out=scores)

    # 6. apply loose nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)

    if pre_nms_count > 0:
        proposals = proposals[:pre_nms_count, :]
        scores = scores[:pre_nms_count]

    if nms_threshold > 0:
        proposals, scores = nms(proposals, scores, nms_threshold)

    if post_nms_count > 0:
        proposals = proposals[:post_nms_count, :]
        scores = scores[:post_nms_count]

    proposals_num = proposals.shape[0]
    if force_max_output_size and proposals_num < post_nms_count:
        fake_proposals = proposals.new_zeros((post_nms_count - proposals_num, 4))
        fake_proposals[:, 2:] = 1.0
        proposals = torch.cat((proposals, fake_proposals), dim=0)
        scores = torch.cat((scores, scores.new_zeros((post_nms_count - proposals_num,))), dim=0)

    return proposals, scores


def filter_boxes(boxes, min_size, im_info):
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    ws = boxes[:, 2] - x0 + 1
    hs = boxes[:, 3] - y0 + 1
    keep = torch.nonzero((ws >= min_size) & (hs >= min_size)).squeeze()
    return keep


class GenerateProposalsSingleImage(torch.autograd.Function):
    @staticmethod
    def symbolic(g, im_info, all_anchors, bbox_deltas, scores,
                 pre_nms_count=12000, post_nms_count=2000, nms_threshold=0.7, min_size=0, force_max_output_size=False):
        return g.op("ExperimentalDetectronGenerateProposalsSingleImage", im_info, all_anchors, bbox_deltas, scores,
                    pre_nms_count_i=pre_nms_count, post_nms_count_i=post_nms_count, nms_threshold_f=nms_threshold,
                    min_size_f=min_size, outputs=2)

    @staticmethod
    def forward(ctx, im_info, all_anchors, bbox_deltas, scores,
                pre_nms_count=12000, post_nms_count=2000, nms_threshold=0.7, min_size=0, force_max_output_size=False):
        return generate_proposals_single_image(im_info, all_anchors, bbox_deltas, scores,
                                               pre_nms_count, post_nms_count, nms_threshold, min_size,
                                               force_max_output_size)
