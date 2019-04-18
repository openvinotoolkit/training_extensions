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
import torch.nn as nn

from ..extensions._EXTRA import roi_align_forward, roi_align_backward


class ROIAlignFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, features, rois, aligned_height=7, aligned_width=7, spatial_scale=1 / 16, sampling_ratio=0):
        return g.op('ExperimentalDetectronROIAlign', features, rois, aligned_height_i=aligned_height,
                    aligned_width_i=aligned_width,
                    spatial_scale_f=spatial_scale, sampling_ratio_i=sampling_ratio)

    @staticmethod
    def forward(ctx, features, rois, aligned_height=7, aligned_width=7, spatial_scale=1 / 16, sampling_ratio=0):

        if isinstance(rois, list) or isinstance(rois, tuple):
            device = features.device
            rois_blob = torch.empty((0, 5), dtype=torch.float32, device=device)
            for i, im_rois in enumerate(rois):
                rois_blob = torch.cat((rois_blob,
                                       torch.cat((torch.full((len(im_rois), 1), i, dtype=torch.float32, device=device),
                                                  im_rois), dim=1)
                                       ), dim=0)
        else:
            rois_blob = rois
        feature_size = features.size()

        ctx.feature_size = feature_size
        ctx.aligned_height = aligned_height
        ctx.aligned_width = aligned_width
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio

        ctx.save_for_backward(rois_blob)

        output_blob = roi_align_forward(features, rois_blob, spatial_scale,
                                        aligned_height, aligned_width, sampling_ratio)
        return output_blob

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError

        rois, = ctx.saved_variables

        assert (ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_blob = roi_align_backward(grad_output, rois, ctx.spatial_scale,
                                       ctx.aligned_height, ctx.aligned_width,
                                       batch_size, num_channels, data_height, data_width,
                                       ctx.sampling_ratio)

        return grad_blob, None, None, None, None, None


def roi_align(features, rois, aligned_height=7, aligned_width=7, spatial_scale=1 / 16, sampling_ratio=0):
    return ROIAlignFunction.apply(features, rois, aligned_height, aligned_width, spatial_scale, sampling_ratio)


class ROIAlign(nn.Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super().__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        return roi_align(features, rois, aligned_height=self.aligned_height, aligned_width=self.aligned_width,
                         spatial_scale=self.spatial_scale, sampling_ratio=self.sampling_ratio)
