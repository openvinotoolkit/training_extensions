from __future__ import annotations
from torchvision.models.detection.mask_rcnn import MaskRCNN

from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
import torch
from collections import OrderedDict
from torchvision.models.detection.image_list import ImageList


class OTXTVMaskRCNN(MaskRCNN):
    def forward(
        self,
        entity: InstanceSegBatchDataEntity,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Overwrite GeneralizedRCNN forward method to accept InstanceSegBatchDataEntity."""
        ori_shapes = [img_info.ori_shape for img_info in entity.imgs_info]
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]

        image_list = ImageList(entity.images, img_shapes)
        targets = []
        for bboxes, labels, masks, polygons in zip(
            entity.bboxes,
            entity.labels,
            entity.masks,
            entity.polygons,
        ):
            targets.append(
                {
                    "boxes": bboxes,
                    # TODO: I think we should do num_classes + 1 (BG) as torchvision.MaskRCNN assume background class?
                    "labels": labels + 1,
                    "masks": masks,
                    "polygons": polygons,
                },
            )

        features = self.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(image_list, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)

        # TODO: check postprocess method (I removed it but still inherited from GeneralizedRCNN)
        detections = self.transform.postprocess(detections, image_list.image_sizes, ori_shapes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)
