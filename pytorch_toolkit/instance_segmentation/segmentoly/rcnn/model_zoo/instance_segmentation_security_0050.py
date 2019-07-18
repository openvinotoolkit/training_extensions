import torch.nn as nn

from .fpn_mask_rcnn_base import (MaskHead, BboxHead, PriorBox, DetectionOutput,
                                 RPN, FPN, ProposalGTMatcher)
from .resnet_fpn_mask_rcnn import ResNet50FPNMaskRCNN
from ..panet import BottomUpPathAugmentation
from ...utils.weights import xavier_fill


class BboxHead3FC(BboxHead):
    def __init__(self, dim_in, dim_out, resolution_in, cls_num,
                 cls_agnostic_bbox_regression=False, fc_as_conv=False, **kwargs):
        super().__init__(dim_in, dim_out, resolution_in, cls_num, cls_agnostic_bbox_regression, fc_as_conv, **kwargs)
        if fc_as_conv:
            self.fc3 = nn.Conv2d(dim_out, dim_out, 1)
        else:
            self.fc3 = nn.Linear(dim_out, dim_out)

        xavier_fill(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        if isinstance(self.fc1, nn.Linear):
            batch_size = int(x.size(0))
            x = x.view(batch_size, -1)
        x = nn.functional.relu(self.fc1(x), inplace=True)
        x = nn.functional.relu(self.fc2(x), inplace=True)
        x = nn.functional.relu(self.fc3(x), inplace=True)
        return self.get_score_and_prediction(x)


class MaskHeadBN(MaskHead):
    def __init__(self, dim_in, num_convs, num_cls, dim_internal=256, dilation=2, **kwargs):
        super().__init__(dim_in, num_convs, num_cls, dim_internal, dilation, **kwargs)
        self.dim_in = dim_in
        self.num_convs = num_convs
        self.dim_out = dim_internal

        del self.conv_fcn
        module_list = []
        for i in range(num_convs):
            module_list.extend([
                nn.Conv2d(dim_in, dim_internal, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(dim_internal),
                nn.ReLU(inplace=True)
            ])
            dim_in = dim_internal
        self.conv_fcn = nn.Sequential(*module_list)

        self.upconv = nn.ConvTranspose2d(dim_internal, dim_internal, kernel_size=2, stride=2, padding=0)
        self.segm = nn.Conv2d(dim_internal, num_cls, 1, 1, 0)

        self._init_weights()


class BottomUpPathAugmentationBN(BottomUpPathAugmentation):

    @staticmethod
    def _conv2d_block(dim_in, dim_out, kernel, stride, padding, bias):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )


class RPNLite(RPN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.conv
        self.conv = nn.Conv2d(self.dim_in, self.dim_internal, 1, 1, 0)

        self.init_weights()


class InstanceSegmentationSecurity0050(ResNet50FPNMaskRCNN):
    segmentation_roi_featuremap_resolution = 7

    def __init__(self, cls_num, **kwargs):
        super().__init__(cls_num, **kwargs)

        self.bupa = BottomUpPathAugmentationBN(output_levels=5, dims_in=self.fpn.dims_out,
                                               scales_in=self.fpn.scales_out, dim_out=128, group_norm=False)

        r = self.segmentation_roi_featuremap_resolution
        self.proposal_gt_matcher = ProposalGTMatcher(positive_threshold=0.5, negative_threshold=0.5,
                                                     positive_fraction=0.25, batch_size=256,
                                                     target_mask_size=(2 * r, 2 * r))

    @staticmethod
    def add_fpn(dims_in, scales_in, **kwargs):
        return FPN(dims_in, scales_in, 128, 128, group_norm=False)

    @staticmethod
    def add_priors_generator():
        prior_boxes = nn.ModuleList()
        widths = [[18.00461443977467 + 1, 28.899706148116053 + 1, 73.42073611573349 + 1],
                  [54.733722536651804 + 1, 136.73193415104578 + 1, 68.17445780221439 + 1],
                  [99.90277461137683 + 1, 155.060481177927 + 1, 254.14803078894226 + 1],
                  [143.22154707415015 + 1, 243.36151152306752 + 1, 416.3044920416812 + 1],
                  [279.57615011598125 + 1, 410.15991631559905 + 1, 450.9415967854453 + 1]]

        heights = [[22.358642713410788 + 1, 61.90766647668873 + 1, 46.83872274215618 + 1],
                   [113.48241169265263 + 1, 89.23900340394223 + 1, 183.0552472694527 + 1],
                   [280.45049255862403 + 1, 185.32713748322118 + 1, 123.3327609158835 + 1],
                   [390.62539933985994 + 1, 270.83141023619 + 1, 163.83324079051005 + 1],
                   [414.2871418830516 + 1, 308.4935963584353 + 1, 445.5373059473707 + 1]]

        scale_factor = 1.0
        for ws, hs in zip(widths, heights):
            if scale_factor != 1.0:
                for i in range(len(ws)):
                    ws[i] *= scale_factor
                for i in range(len(hs)):
                    hs[i] *= scale_factor
            prior_boxes.append(PriorBox(widths=ws, heights=hs, flatten=True, use_cache=True))
        priors_per_level_num = list([priors.priors_num() for priors in prior_boxes])
        assert priors_per_level_num[1:] == priors_per_level_num[:-1]
        priors_num = priors_per_level_num[0]
        return prior_boxes, priors_num

    @staticmethod
    def add_segmentation_head(features_dim_in, cls_num, **kwargs):
        # ROI-wise segmentation part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        mask_head = MaskHeadBN(features_dim_in[0], 6, cls_num, 128, 1)
        return mask_head

    @staticmethod
    def add_rpn(priors_num, features_dim_in):
        # RPN is shared between FPN levels.
        assert features_dim_in[1:] == features_dim_in[:-1]
        rpn = RPNLite(features_dim_in[0], 128, priors_num, 'sigmoid')
        return rpn

    @staticmethod
    def add_detection_head(features_dim_in, cls_num, fc_detection_head=True, **kwargs):
        # ROI-wise detection part.
        assert features_dim_in[1:] == features_dim_in[:-1]
        dim_out = 512
        detection_head = BboxHead3FC(features_dim_in[0], dim_out, 7, cls_num,
                                     cls_agnostic_bbox_regression=False,
                                     fc_as_conv=not fc_detection_head)
        detection_output = DetectionOutput(cls_num, nms_threshold=0.5, score_threshold=0.05, post_nms_count=100)
        return detection_head, detection_output

    @property
    def pre_nms_rois_count(self):
        return 2000 if self.training else 100

    @property
    def post_nms_rois_count(self):
        return 2000 if self.training else 100

    def forward_fpn(self, feature_pyramid):
        x = self.fpn(feature_pyramid)
        return self.bupa(x)
