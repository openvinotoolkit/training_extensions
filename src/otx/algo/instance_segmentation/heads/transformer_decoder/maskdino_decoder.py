# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""MaskDINODecoder module."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.utils.utils import inverse_sigmoid
from otx.algo.instance_segmentation.heads.transformer_decoder.dino_decoder import (
    DeformableTransformerDecoderLayer,
    TransformerDecoder,
)
from otx.algo.instance_segmentation.utils import box_ops
from otx.algo.instance_segmentation.utils.utils import (
    MLP,
    gen_encoder_output_proposals,
)


class MaskDINODecoder(nn.Module):
    """MaskDINODecoder module.

    Args:
        num_classes: number of classes
        hidden_dim: Transformer feature dimension
        num_queries: number of queries
        nheads: number of heads
        dim_feedforward: feature dimension in feedforward network
        dec_layers: number of Transformer decoder layers
        mask_dim: mask feature dimension
        noise_scale: noise scale
        dn_num: number of denoising queries
        total_num_feature_levels: total number of feature levels
        dropout: dropout rate
        nhead: num heads in multi-head attention
        dec_n_points: number of sampling points in decoder
        query_dim: 4 -> (x, y, w, h)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        mask_dim: int,
        noise_scale: float,
        dn_num: int,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        nhead: int = 8,
        dec_n_points: int = 4,
        query_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_feature_levels = total_num_feature_levels

        # define Transformer decoder here
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.total_num_feature_levels = total_num_feature_levels

        self.num_queries = num_queries
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential())
        self.num_classes = num_classes
        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            dim_feedforward,
            dropout,
            self.num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            decoder_norm,
            d_model=hidden_dim,
            query_dim=query_dim,
        )

        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

    def prepare_for_dn(
        self,
        targets: list[dict[str, Tensor]],
        batch_size: int,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, dict[str, Tensor] | None]:
        """Prepare the input for denoising training."""
        if self.training:
            scalar, noise_scale = self.dn_num, self.noise_scale

            known = [(torch.ones_like(t["labels"])) for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            scalar = scalar // int(max(known_num))
            if scalar == 0:
                return None, None, None, None

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t["labels"] for t in targets])
            boxes = torch.cat([t["boxes"] for t in targets])
            batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            p = torch.rand_like(torch.tensor(known_labels_expaned, dtype=known_bbox_expand.dtype))
            chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0), diff) * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long()
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim, device=m.device)
            padding_bbox = torch.zeros(pad_size, 4, device=m.device)

            input_query_label = padding_label.repeat(batch_size, 1, 1)
            input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([], device=m.device)
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size, device=m.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True
                else:
                    attn_mask[single_pad * i : single_pad * (i + 1), single_pad * (i + 1) : pad_size] = True
                    attn_mask[single_pad * i : single_pad * (i + 1), : single_pad * i] = True
            mask_dict = {
                "known_indice": torch.as_tensor(known_indice).long(),
                "batch_idx": torch.as_tensor(batch_idx).long(),
                "map_known_indice": torch.as_tensor(map_known_indice).long(),
                "known_lbs_bboxes": (known_labels, known_bboxs),
                "know_idx": know_idx,
                "pad_size": pad_size,
                "scalar": scalar,
            }
            return input_query_label, input_query_bbox, attn_mask, mask_dict
        return None, None, None, None

    def dn_post_process(
        self,
        outputs_class: Tensor,
        outputs_coord: Tensor,
        mask_dict: dict[str, Tensor],
        outputs_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Post process of dn after output from the transformer put the dn part in the mask_dict."""
        output_known_class = outputs_class[:, :, : mask_dict["pad_size"], :]
        outputs_class = outputs_class[:, :, mask_dict["pad_size"] :, :]
        output_known_coord = outputs_coord[:, :, : mask_dict["pad_size"], :]
        outputs_coord = outputs_coord[:, :, mask_dict["pad_size"] :, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, : mask_dict["pad_size"], :]
            outputs_mask = outputs_mask[:, :, mask_dict["pad_size"] :, :]
        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
            "pred_masks": output_known_mask[-1],
        }

        out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_mask, output_known_coord)
        mask_dict["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Get the valid ratio of the mask."""
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_height / height
        valid_ratio_w = valid_width / width
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def pred_box(self, reference: Tensor, hs: list[Tensor], ref0: Tensor | None = None) -> Tensor:
        """Predict boxes."""
        device = reference[0].device
        outputs_coord_list = [] if ref0 is None else [ref0.to(device)]
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs, strict=True):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        return torch.stack(outputs_coord_list)

    def forward(
        self,
        x: list[Tensor],
        mask_features: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        """Forward pass."""
        if len(x) != self.num_feature_levels:
            msg = "Input feature levels should be equal to the number of feature levels"
            raise ValueError(msg)
        device = x[0].device
        masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=device, dtype=torch.bool) for src in x]
        src_flatten, mask_flatten, spatial_shapes = self.flatten_and_concat_features(x, masks, device)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected = self.class_embed(output_memory)
        enc_outputs_coord_unselected = self._bbox_embed(output_memory) + output_proposals

        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoid
        refpoint_embed = refpoint_embed_undetach.detach()

        tgt_undetach = torch.gather(
            output_memory,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim),
        )  # unsigmoid

        outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
        tgt = tgt_undetach.detach()
        interm_outputs = {}
        interm_outputs["pred_logits"] = outputs_class
        interm_outputs["pred_boxes"] = refpoint_embed_undetach.sigmoid()
        interm_outputs["pred_masks"] = outputs_mask

        # convert masks into boxes to better initialize box in the decoder
        flaten_mask = outputs_mask.detach().flatten(0, 1)
        height, width = outputs_mask.shape[-2:]
        refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0, dtype=flaten_mask.dtype)
        refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.tensor(
            [width, height, width, height],
            dtype=flaten_mask.dtype,
            device=device,
        )
        refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
        refpoint_embed = inverse_sigmoid(refpoint_embed)

        tgt_mask = None
        mask_dict = None
        if self.training:
            if targets is None:
                msg = "In training mode, targets should be passed"
                raise ValueError(msg)
            input_query_label, input_query_bbox, tgt_mask, mask_dict = self.prepare_for_dn(
                targets,
                x[0].shape[0],
            )
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)

        # direct prediction from the matching and denoising part in the begining
        outputs_class, outputs_mask = self.forward_prediction_heads(
            tgt.transpose(0, 1),
            mask_features,
            self.training,
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        if self.training and mask_dict is not None:
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
        )
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1),
                mask_features,
                self.training or (i == len(hs) - 1),
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # iteratively box prediction
        out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())

        if mask_dict is not None:
            predictions_mask = torch.stack(predictions_mask)
            predictions_class = torch.stack(predictions_class)
            predictions_class, out_boxes, predictions_mask = self.dn_post_process(
                predictions_class,
                out_boxes,
                mask_dict,
                predictions_mask,
            )
            predictions_class, predictions_mask = list(predictions_class), list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0 * self.label_enc.weight.sum()

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": out_boxes[-1],
        }

        # Add auxiliary outputs and intermediate output in training for loss computation.
        if self.training:
            out.update(
                {
                    "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, out_boxes),
                    "interm_outputs": interm_outputs,
                },
            )
        return out, mask_dict

    def flatten_and_concat_features(
        self,
        x: list[Tensor],
        masks: list[Tensor],
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Flatten and concatenate features."""
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        return (
            torch.cat(src_flatten, 1),
            torch.cat(mask_flatten, 1),
            torch.as_tensor(spatial_shapes, dtype=torch.long, device=device),
        )

    def forward_prediction_heads(
        self,
        output: Tensor,
        mask_features: Tensor,
        pred_mask: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Forward prediction heads."""
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(
        self,
        outputs_class: Tensor,
        outputs_seg_masks: Tensor,
        out_boxes: Tensor,
    ) -> list[dict[str, Tensor]]:
        return [
            {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
            for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1], strict=True)
        ]
