# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

import logging as log
import pickle  # nosec  B403   used pickle for dumping object
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Literal

import torch
import torchvision.transforms.v2 as tvt_v2
from torch import Tensor, nn
from torchvision.tv_tensors import BoundingBoxes, Image, Mask

from otx.algo.visual_prompting.decoders import SAMMaskDecoder
from otx.algo.visual_prompting.encoders import SAMImageEncoder, SAMPromptEncoder
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from otx.algo.visual_prompting.visual_prompters import SegmentAnything, ZeroShotSegmentAnything
from otx.core.data.entity.base import OTXBatchLossEntity, Points
from otx.core.data.entity.visual_prompting import (
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingBatchPredEntity,
)
from otx.core.metrics.visual_prompting import VisualPromptingMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.visual_prompting import OTXVisualPromptingModel, OTXZeroShotVisualPromptingModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes, NullLabelInfo

if TYPE_CHECKING:
    import numpy as np
    from datumaro import Polygon as dmPolygon
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class CommonSettingMixin:
    """Mixin class for common settings in SAM.

    Attributes:
        model (nn.Module): The model used in SAM.
        load_from (ClassVar[dict[str, str]]): A dictionary containing the URLs to load checkpoints from.

    """

    model: nn.Module
    load_from: ClassVar[dict[str, str]] = {
        "tiny_vit": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    load_state_dict: Callable[[dict[str, Tensor]], None]

    def load_checkpoint(self, load_from: str | None) -> None:
        """Load checkpoint for SAM.

        Args:
            load_from (Optional[str], optional): Checkpoint path for SAM. Defaults to None.
        """
        try:
            state_dict = torch.hub.load_state_dict_from_url(str(load_from))
            for key in [
                "image_encoder.norm_head.weight",
                "image_encoder.norm_head.bias",
                "image_encoder.head.weight",
                "image_encoder.head.bias",
            ]:
                if key in state_dict:
                    state_dict.pop(key)

            # add prefix 'model.' to all keys
            for key in list(state_dict.keys()):
                state_dict["model." + key] = state_dict.pop(key)

            self.load_state_dict(state_dict)

        except (ValueError, RuntimeError) as e:
            log.info(
                f"{e}: {load_from} is not desirable format for torch.hub.load_state_dict_from_url. "
                f"To manually load {load_from}, try to set it to trainer.checkpoint.",
            )

    def freeze_networks(
        self,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        """Freeze networks depending on config.

        Args:
            freeze_image_encoder (bool): Whether to freeze the image encoder.
            freeze_prompt_encoder (bool): Whether to freeze the prompt encoder.
            freeze_mask_decoder (bool): Whether to freeze the mask decoder.
        """
        for param in self.model.image_encoder.parameters():
            param.requires_grad = not freeze_image_encoder

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = not freeze_prompt_encoder

        for param in self.model.mask_decoder.parameters():
            param.requires_grad = not freeze_mask_decoder

    @torch.no_grad()
    def forward_for_tracing(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        ori_shape: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward method for SAM inference (export/deploy).

        Args:
            image_embeddings (Tensor): The image embedding with a batch index of length 1.
                If it is a zero tensor, the image embedding will be computed from the image.
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs.
                Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.
            mask_input (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.
                This input has 1x1 shape due to supporting openvino input layout.
            ori_shape (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.
        """
        return self.model.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )


class SAM(OTXVisualPromptingModel, CommonSettingMixin):
    """OTX visual prompting model class for Segment Anything Model (SAM)."""

    input_size_multiplier = 16

    def __init__(
        self,
        backbone_type: Literal["tiny_vit", "vit_b"],
        label_info: LabelInfoTypes = NullLabelInfo(),
        input_size: tuple[int, int] = (1024, 1024),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        use_stability_score: bool = False,
        return_single_mask: bool = True,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        if input_size[0] != input_size[1]:
            msg = f"SAM should use square image size, but got {input_size}"
            raise ValueError(msg)

        self.backbone_type = backbone_type
        self.image_size = input_size[0]
        self.image_embedding_size = input_size[0] // 16

        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        self.load_checkpoint(load_from=self.load_from[backbone_type])
        self.freeze_networks(freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder)

    def _build_model(self) -> nn.Module:
        image_encoder = SAMImageEncoder(backbone_type=self.backbone_type, img_size=self.image_size)
        prompt_encoder = SAMPromptEncoder(
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
        )
        mask_decoder = SAMMaskDecoder()
        criterion = SAMCriterion(image_size=self.image_size)
        return SegmentAnything(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            criterion=criterion,
            image_size=self.image_size,
            use_stability_score=self.use_stability_score,
            return_single_mask=self.return_single_mask,
            return_extra_metrics=self.return_extra_metrics,
            stability_score_offset=self.stability_score_offset,
        )


class ZeroShotSAM(OTXZeroShotVisualPromptingModel, CommonSettingMixin):
    """Zero-Shot Visual Prompting model."""

    def __init__(  # noqa: PLR0913
        self,
        backbone_type: Literal["tiny_vit", "vit_b"],
        label_info: LabelInfoTypes = NullLabelInfo(),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
        reference_info_dir: Path | str = "reference_infos",
        infer_reference_info_root: Path | str = "../.latest/train",
        save_outputs: bool = True,
        pixel_mean: list[float] | None = [123.675, 116.28, 103.53],  # noqa: B006
        pixel_std: list[float] | None = [58.395, 57.12, 57.375],  # noqa: B006
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = True,
        default_threshold_reference: float = 0.3,
        default_threshold_target: float = 0.65,
        use_stability_score: bool = False,
        return_single_mask: bool = False,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        self.backbone_type = backbone_type
        self.image_size = 1024  # zero-shot visual prompting model uses fixed 1024x1024 input size
        self.image_embedding_size = 1024 // 16  # zero-shot visual prompting model uses fixed 1024x1024 input size

        self.default_threshold_reference = default_threshold_reference
        self.default_threshold_target = default_threshold_target

        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset

        super().__init__(
            label_info=label_info,
            input_size=(1024, 1024),  # zero-shot visual prompting model uses fixed 1024x1024 input size
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        # check freeze conditions
        if not (freeze_image_encoder and freeze_prompt_encoder and freeze_mask_decoder):
            log.warning(
                "All of freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder "
                "must be set to True, changed.",
            )
            freeze_image_encoder = True
            freeze_prompt_encoder = True
            freeze_mask_decoder = True

        self.load_checkpoint(load_from=self.load_from[backbone_type])
        self.freeze_networks(freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder)

        self.save_outputs = save_outputs
        self.reference_info_dir: Path = Path(reference_info_dir)
        self.infer_reference_info_root: Path = Path(infer_reference_info_root)

        self.register_buffer("pixel_mean", Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", Tensor(pixel_std).view(-1, 1, 1), False)

        self.initialize_reference_info()

    def _build_model(self) -> nn.Module:
        image_encoder = SAMImageEncoder(backbone_type=self.backbone_type, img_size=self.image_size)
        prompt_encoder = SAMPromptEncoder(
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
        )
        mask_decoder = SAMMaskDecoder()
        criterion = None
        return ZeroShotSegmentAnything(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            criterion=criterion,
            image_size=self.image_size,
            default_threshold_reference=self.default_threshold_reference,
            default_threshold_target=self.default_threshold_target,
            use_stability_score=self.use_stability_score,
            return_single_mask=self.return_single_mask,
            return_extra_metrics=self.return_extra_metrics,
            stability_score_offset=self.stability_score_offset,
        )

    def forward(  # type: ignore[override]
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,  # type: ignore[override]
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        forward_fn = self.learn if self.training else self.infer
        return forward_fn(inputs)  # type: ignore[operator]

    def learn(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
        reset_feat: bool = False,
        is_cascade: bool = False,
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Learn to directly connect to the model."""
        self.training = True
        if reset_feat:
            self.initialize_reference_info()

        outputs = self.model.learn(
            **self._customize_inputs(inputs, reference_feats=reference_feats, used_indices=used_indices),
            is_cascade=is_cascade,
        )
        return self._customize_outputs(outputs, inputs)

    def infer(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
        reference_feats: Tensor | None = None,
        used_indices: Tensor | None = None,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        is_cascade: bool = True,
    ) -> ZeroShotVisualPromptingBatchPredEntity | OTXBatchLossEntity:
        """Infer to directly connect to the model."""
        self.training = False
        outputs = self.model.infer(
            **self._customize_inputs(inputs, reference_feats=reference_feats, used_indices=used_indices),
            threshold=threshold,
            num_bg_points=num_bg_points,
            is_cascade=is_cascade,
        )
        return self._customize_outputs(outputs, inputs)

    def _gather_prompts_with_labels(
        self,
        inputs: ZeroShotVisualPromptingBatchDataEntity,
    ) -> list[dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]]:
        """Gather prompts according to labels."""
        total_processed_prompts: list[dict[int, list[BoundingBoxes | Points | dmPolygon | Mask]]] = []
        for batch, batch_labels in enumerate(inputs.labels):
            processed_prompts = defaultdict(list)
            for prompt_type in ["prompts", "polygons", "masks"]:
                _prompts = getattr(inputs, prompt_type, None)
                prompt_labels = getattr(batch_labels, prompt_type, None)
                if _prompts is None or prompt_labels is None:
                    continue

                for idx, _label in enumerate(prompt_labels):
                    if prompt_type in ("prompts", "polygons"):
                        processed_prompts[int(_label)].append(_prompts[batch][idx])
                    else:
                        # for mask
                        processed_prompts[int(_label)].append(Mask(_prompts[batch][idx]))

            sorted_processed_prompts = dict(sorted(processed_prompts.items(), key=lambda x: x))
            total_processed_prompts.append(sorted_processed_prompts)

        return total_processed_prompts

    def apply_image(self, image: Image | np.ndarray, target_length: int = 1024) -> Image:
        """Preprocess image to be used in the model."""
        h, w = image.shape[-2:]
        target_size = self.get_preprocess_shape(h, w, target_length)
        return tvt_v2.functional.resize(tvt_v2.functional.to_image(image), target_size, antialias=True)

    def apply_coords(self, coords: Tensor, ori_shape: tuple[int, ...], target_length: int = 1024) -> Tensor:
        """Preprocess points to be used in the model."""
        old_h, old_w = ori_shape
        new_h, new_w = self.get_preprocess_shape(ori_shape[0], ori_shape[1], target_length)
        coords = deepcopy(coords).to(torch.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_points(self, points: Points, ori_shape: tuple[int, ...], target_length: int = 1024) -> Points:
        """Preprocess points to be used in the model."""
        return Points(self.apply_coords(points, ori_shape, target_length), canvas_size=(target_length, target_length))

    def apply_boxes(self, boxes: BoundingBoxes, ori_shape: tuple[int, ...], target_length: int = 1024) -> BoundingBoxes:
        """Preprocess boxes to be used in the model."""
        return BoundingBoxes(
            self.apply_coords(boxes.reshape(-1, 2, 2), ori_shape, target_length).reshape(-1, 4),
            format=boxes.format,
            canvas_size=(target_length, target_length),
        )

    def apply_prompts(
        self,
        prompts: list[Points | BoundingBoxes],
        ori_shape: tuple[int, ...],
        target_length: int = 1024,
    ) -> list[Points | BoundingBoxes]:
        """Preprocess prompts to be used in the model."""
        transformed_prompts: list[Points | BoundingBoxes] = []
        for prompt in prompts:
            if isinstance(prompt, Points):
                transformed_prompts.append(self.apply_points(prompt, ori_shape, target_length))
            elif isinstance(prompt, BoundingBoxes):
                transformed_prompts.append(self.apply_boxes(prompt, ori_shape, target_length))
            else:
                log.info(f"Current prompt ({prompt.__class__.__name__}) is not supported, saved as it is.")
                transformed_prompts.append(prompt)
        return transformed_prompts

    def get_preprocess_shape(self, oldh: int, oldw: int, target_length: int) -> tuple[int, int]:
        """Get preprocess shape."""
        scale = target_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess(self, x: Image) -> Image:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        x = self.model.pad_to_square(x)
        return Image(x)

    def transforms(self, entity: ZeroShotVisualPromptingBatchDataEntity) -> ZeroShotVisualPromptingBatchDataEntity:
        """Transforms for ZeroShotVisualPromptingBatchDataEntity."""
        return entity.wrap(
            images=[self.preprocess(self.apply_image(image)) for image in entity.images],
            prompts=[
                self.apply_prompts(prompt, info.ori_shape, self.model.image_size)
                for prompt, info in zip(entity.prompts, entity.imgs_info)
            ],
            masks=entity.masks,
            polygons=entity.polygons,
            labels=entity.labels,
        )

    def initialize_reference_info(self) -> None:
        """Initialize reference information."""
        self.register_buffer("reference_feats", torch.zeros(0, 1, self.model.prompt_encoder.embed_dim), False)
        self.register_buffer("used_indices", torch.tensor([], dtype=torch.int64), False)

    def save_reference_info(self, default_root_dir: Path | str) -> None:
        """Save reference info."""
        reference_info = {
            "reference_feats": self.reference_feats,
            "used_indices": self.used_indices,
        }
        # save reference info
        self.saved_reference_info_path: Path = Path(default_root_dir) / self.reference_info_dir / "reference_info.pt"
        self.saved_reference_info_path.parent.mkdir(parents=True, exist_ok=True)
        # TODO (sungchul): ticket no. 139210
        torch.save(reference_info, self.saved_reference_info_path)
        pickle.dump(
            {k: v.numpy() for k, v in reference_info.items()},
            self.saved_reference_info_path.with_suffix(".pickle").open("wb"),
        )
        log.info(f"Saved reference info at {self.saved_reference_info_path}.")

    def load_reference_info(
        self,
        default_root_dir: Path | str,
        device: str | torch.device = "cpu",
        path_to_directly_load: Path | None = None,
    ) -> bool:
        """Load latest reference info to be used.

        Args:
            default_root_dir (Path | str): Default root directory to be used
                when inappropriate infer_reference_info_root is given.
            device (str | torch.device): Device that reference infos will be attached.
            path_to_directly_load (Path | None): Reference info path to directly be loaded.
                Normally, it is obtained after `learn` which is executed when trying to do `infer`
                without reference features in `on_test_start` or `on_predict_start`.

        Returns:
            (bool): Whether normally loading checkpoint or not.
        """
        if path_to_directly_load is not None:
            # if `path_to_directly_load` is given, forcely load
            reference_info = torch.load(path_to_directly_load)
            retval = True
            log.info(f"reference info saved at {path_to_directly_load} was successfully loaded.")

        else:
            if str(self.infer_reference_info_root) == "../.latest/train":
                # for default setting
                path_reference_info = (
                    Path(default_root_dir)
                    / self.infer_reference_info_root
                    / self.reference_info_dir
                    / "reference_info.pt"
                )
            else:
                # for user input
                path_reference_info = self.infer_reference_info_root / self.reference_info_dir / "reference_info.pt"

            if path_reference_info.is_file():
                reference_info = torch.load(path_reference_info)
                retval = True
                log.info(f"reference info saved at {path_reference_info} was successfully loaded.")
            else:
                reference_info = {}
                retval = False

        self.register_buffer(
            "reference_feats",
            reference_info.get("reference_feats", torch.zeros(0, 1, self.model.prompt_encoder.embed_dim)).to(device),
            False,
        )
        self.register_buffer(
            "used_indices",
            reference_info.get("used_indices", torch.tensor([], dtype=torch.int64)).to(device),
            False,
        )
        return retval
