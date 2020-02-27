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
from torch import nn

from nncf.utils import no_jit_trace
from nncf.dynamic_graph.context import no_nncf_trace

from ..box_utils import decode, nms


class DetectionOutput(nn.Module):
    def __init__(self, num_classes, background_label_id, top_k, keep_top_k, confidence_threshold, nms_threshold,
                 eta=1, share_location=1, code_type='CENTER_SIZE', variance_encoded_in_target=0):
        super().__init__()
        self.num_classes = num_classes
        self.background_label_id = background_label_id
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.eta = eta
        self.share_location = share_location
        self.code_type = code_type
        self.variance_encoded_in_target = variance_encoded_in_target

    def forward(self, loc_data, conf_data, prior_data):
        return DetectionOutputFunction.apply(loc_data, conf_data, prior_data, self)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


class DetectionOutputFunction(torch.autograd.Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    @staticmethod
    def symbolic(g, loc_data, conf_data, prior_data, detection_output_params):
        return g.op("DetectionOutput", loc_data, conf_data, prior_data,
                    num_classes_i=detection_output_params.num_classes,
                    background_label_id_i=detection_output_params.background_label_id,
                    top_k_i=detection_output_params.top_k,
                    keep_top_k_i=detection_output_params.keep_top_k,
                    confidence_threshold_f=detection_output_params.confidence_threshold,
                    nms_threshold_f=detection_output_params.nms_threshold, eta_f=detection_output_params.eta,
                    share_location_i=detection_output_params.share_location,
                    code_type_s=detection_output_params.code_type,
                    variance_encoded_in_target_i=detection_output_params.variance_encoded_in_target)

    @staticmethod
    def forward(ctx, loc_data, conf_data, prior_data, detection_output_params):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors*num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,2,num_priors*4]
        """
        with no_jit_trace(), no_nncf_trace():
            if detection_output_params.nms_threshold <= 0:
                raise ValueError('nms_threshold must be non negative.')
            device = loc_data.device
            batch_size = loc_data.size(0)  # batch size
            num_priors = int(loc_data.size(1) / 4)
            loc_data = loc_data.view(batch_size, num_priors, 4)
            conf_data = conf_data.view(batch_size, num_priors, -1)
            prior_data = prior_data.view(1, 2, num_priors, 4)
            output = torch.zeros(batch_size, 1, detection_output_params.keep_top_k, 7).to(device)

            conf_preds = conf_data.view(batch_size, num_priors,
                                        detection_output_params.num_classes).transpose(2, 1)

            # Decode predictions into bboxes.
            for i in range(batch_size):
                output_for_img = torch.zeros(0, 7).to(device)
                decoded_boxes = decode(loc_data[i], prior_data[0])
                # For each class, perform nms
                conf_scores = conf_preds[i].clone()

                total_detections_count = 0
                all_indices = dict()  # indices of confident detections for each class
                boxes = dict()
                for cl in range(0, detection_output_params.num_classes):
                    if cl == detection_output_params.background_label_id:
                        continue
                    c_mask = conf_scores[cl].gt(detection_output_params.confidence_threshold)
                    scores = conf_scores[cl][c_mask]
                    if scores.dim() == 0:
                        continue
                    conf_scores[cl, :scores.size()[0]] = scores
                    conf_scores[cl, scores.size()[0]:] = 0
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                    boxes[cl] = decoded_boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    all_indices[cl], count = nms(boxes[cl], scores, detection_output_params.nms_threshold,
                                                 detection_output_params.top_k)
                    all_indices[cl] = all_indices[cl][:count]
                    total_detections_count += count

                score_index_pairs = list()  # list of tuples (score, label, idx)
                for label, indices in all_indices.items():
                    indices = indices.cpu().numpy()
                    for idx in indices:
                        score_index_pairs.append((conf_scores[label, idx], label, idx))

                score_index_pairs.sort(key=lambda tup: tup[0], reverse=True)
                score_index_pairs = score_index_pairs[:detection_output_params.keep_top_k]

                all_indices_new = dict()
                for _, label, idx in score_index_pairs:
                    if label not in all_indices_new:
                        all_indices_new[label] = [idx]
                    else:
                        all_indices_new[label].append(idx)

                for label, indices in all_indices_new.items():
                    out = torch.cat((
                        torch.zeros((len(indices), 1), dtype=torch.float).new_full((len(indices), 1), i).to(device),
                        torch.zeros((len(indices), 1), dtype=torch.float).new_full((len(indices), 1), label).to(device),
                        conf_scores[label, indices].unsqueeze(1).to(device),
                        boxes[label][indices].to(device)
                    ), 1)
                    output_for_img = torch.cat((output_for_img, out), 0)

                output[i, 0, :output_for_img.size()[0]] = output_for_img
        return output

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out
