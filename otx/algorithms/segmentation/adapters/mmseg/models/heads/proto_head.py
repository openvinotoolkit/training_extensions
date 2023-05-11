import torch.nn as nn
import torch
import torch.nn.functional as F
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.aspp_head import ASPPHead
from mmseg.ops import resize
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from einops import rearrange, repeat
from otx.algorithms.segmentation.adapters.mmseg.models.utils import distributed_sinkhorn, momentum_update, ProjectionHead, trunc_normal_
import torch.distributed as dist
from  mmseg.models.losses import accuracy


@HEADS.register_module()
class ProtoNet(BaseDecodeHead):
    def __init__(self, gamma, num_prototype, in_proto_channels, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.gamma = gamma
        self.num_prototype = num_prototype
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_proto_channels),
                                requires_grad=False)
        trunc_normal_(self.prototypes, std=0.02)
        self.avg_pool = nn.AdaptiveAvgPool2d(256)
        self.proj_head = ProjectionHead(in_proto_channels, in_proto_channels)
        self.feat_norm = nn.LayerNorm(in_proto_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

    def __init_prototypes(self):
        pass

    def __forward_aspp(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        return aspp_outs

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()

        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  # n x self.num_prototype
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  # n x embedding_dim
            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            n = torch.sum(m_q, dim=0)
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

        self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1),
                                        requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, inputs, gt_semantic_seg, orig_size=(512,512)):
        return self.forward_proto(inputs, gt_semantic_seg, orig_size=(512,512))

    def forward_proto(self, inputs, gt_semantic_seg, orig_size=(512,512)):
        c = self.proj_head(inputs)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = F.normalize(_c, p=2, dim=-1)
        self.prototypes.data.copy_(F.normalize(self.prototypes, p=2, dim=-1))
        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=inputs.shape[0], h=inputs.shape[2])
        gt_seg = F.interpolate(gt_semantic_seg.float(), size=inputs.size()[2:], mode='nearest').view(-1)
        contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
        out_seg =  F.interpolate(out_seg, size=orig_size, mode='bilinear')
        proto_out = {"out_seg": out_seg, "contrast_logits": contrast_logits, "contrast_target": contrast_target}

        return proto_out

    @force_fp32(apply_to=("out_seg", "contrast_logits", "contrast_target",))
    def losses(self, out_seg, contrast_logits, contrast_target, seg_label):
        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    out_seg,
                    contrast_logits,
                    contrast_target,
                    seg_label)
            else:
                loss[loss_decode.loss_name] = loss_decode(
                    out_seg,
                    contrast_logits,
                    contrast_target,
                    seg_label)

        loss['acc_seg'] = accuracy(out_seg, seg_label.squeeze(1), ignore_index=self.ignore_index)

        return loss
