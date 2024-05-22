from __future__ import annotations

import torch
from torch import nn

from .linear_head import LinearClsHead


class OTXSemiSLClsHead:
    """Classification head for Semi-SL.

    Args:
        unlabeled_coef (float): unlabeled loss coefficient, default is 1.0
        dynamic_threshold (boolean): whether to use dynamic threshold, default is True
        min_threshold (float): Minimum value of threshold determining pseudo-label, default is 0.5
    """
    loss_module: nn.Module

    def __init__(
        self,
        num_classes: int,
        unlabeled_coef: float=1.0,
        use_dynamic_threshold: bool=True,
        min_threshold: float=0.5,
    ):
        self.num_classes = num_classes

        self.unlabeled_coef = unlabeled_coef
        self.use_dynamic_threshold = use_dynamic_threshold
        self.min_threshold = (
            min_threshold if self.use_dynamic_threshold else 0.95
        )  # the range of threshold will be [min_thr, 1.0]
        self.num_pseudo_label = 0
        self.classwise_acc = torch.ones((self.num_classes,)) * self.min_threshold

    def loss(self, feats: tuple[torch.Tensor], labels: torch.Tensor, pseudo_label=None, mask=None):
        """Loss function in which unlabeled data is considered.

        Args:
            logits (set): (labeled data logit, unlabeled data logit)
            gt_label (Tensor): target features for labeled data
            pseudo_label (Tensor): target feature for unlabeled data
            mask (Tensor): Mask that shows pseudo-label that passes threshold

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        logits_x, logits_u_s = feats
        num_samples = len(logits_x)
        losses = {}

        # compute supervised loss
        labeled_loss = self.loss_module(logits_x, gt_label, avg_factor=num_samples)

        unlabeled_loss = 0
        if len(logits_u_s) > 0:
            # compute unsupervised loss
            unlabeled_loss = self.loss_module(logits_u_s, pseudo_label, avg_factor=len(logits_u_s)) * mask
        losses["loss"] = labeled_loss + self.unlabeled_coef * unlabeled_loss
        losses["unlabeled_loss"] = self.unlabeled_coef * unlabeled_loss

        return losses

    def forward_train(self, x, gt_label, final_layer=None):
        """Forward_train head using pseudo-label selected through threshold.

        Args:
            x (dict or Tensor): dict(labeled, unlabeled_weak, unlabeled_strong) or NxC input features.
            gt_label (Tensor): NxC target features.
            final_layer (nn.Linear or nn.Sequential): a final layer forwards feature from backbone.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        label_u, mask = None, None
        if isinstance(x, dict):
            for key in x.keys():
                x[key] = self.pre_logits(x[key])
            outputs = final_layer(x["labeled"])  # Logit of Labeled Img
            batch_size = len(outputs)

            with torch.no_grad():
                logit_uw = final_layer(x["unlabeled_weak"])
                pseudo_label = torch.softmax(logit_uw.detach(), dim=-1)
                max_probs, label_u = torch.max(pseudo_label, dim=-1)

                # select Pseudo-Label using flexible threhold
                self.classwise_acc = self.classwise_acc.to(label_u.device)
                mask = max_probs.ge(self.classwise_acc[label_u]).float()
                self.num_pseudo_label = mask.sum()

                if self.use_dynamic_threshold:
                    # get Labeled Data True Positive Confidence
                    logit_x = torch.softmax(outputs.detach(), dim=-1)
                    x_probs, x_idx = torch.max(logit_x, dim=-1)
                    x_probs = x_probs[x_idx == gt_label]
                    x_idx = x_idx[x_idx == gt_label]

                    # get Unlabeled Data Selected Confidence
                    uw_probs = max_probs[mask == 1]
                    uw_idx = label_u[mask == 1]

                    # update class-wise accuracy
                    for i in set(x_idx.tolist() + uw_idx.tolist()):
                        current_conf = torch.tensor([x_probs[x_idx == i].mean(), uw_probs[uw_idx == i].mean()])
                        current_conf = current_conf[~current_conf.isnan()].mean()
                        self.classwise_acc[i] = max(current_conf, self.min_threshold)

            outputs = torch.cat((outputs, final_layer(x["unlabeled_strong"])))
        else:
            outputs = final_layer(x)
            batch_size = len(outputs)

        logits_x = outputs[:batch_size]
        logits_u = outputs[batch_size:]
        del outputs
        logits = (logits_x, logits_u)
        return self.loss(logits, gt_label, label_u, mask)


class OTXSemiSLLinearClsHead(OTXSemiSLClsHead, LinearClsHead):
    """"""
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        topk: int | tuple = (1,),
        init_cfg: dict = {"type": "Normal", "layer": "Linear", "std": 0.01},  # noqa: B006
        unlabeled_coef: float = 1,
        use_dynamic_threshold: bool = True,
        min_threshold: float = 0.5,
    ):
        LinearClsHead.__init__(self, num_classes, in_channels, loss, topk, init_cfg=init_cfg)
        OTXSemiSLClsHead.__init__(self, num_classes, unlabeled_coef, use_dynamic_threshold, min_threshold)

class OTXSemiSLNonLinearClsHead(OTXSemiSLClsHead):
    """"""