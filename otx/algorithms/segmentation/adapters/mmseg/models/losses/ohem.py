import torch
from torch.nn import functional as F
import torch.nn as nn
from mmseg.models.builder import LOSSES
from otx.algorithms.segmentation.adapters.mmseg.models.losses import CrossEntropyLossWithIgnore


@LOSSES.register_module()
class CriterionOhem(nn.Module):
    def __init__(self, aux_weight=0, thresh=0.7, min_kept=100000,  ignore_index=255, **kwargs):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._loss_name = "ohem_loss"
        self._criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept, **kwargs)
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept, **kwargs)

    def forward(self, preds, target, aux=None, **kwargs):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred = preds
            aux_pred = aux
            loss1 = self._criterion1(main_pred, target, **kwargs)
            loss2 = self._criterion2(aux_pred, target, **kwargs)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target, **kwargs)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=256, reduce=False, **kwargs):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = CrossEntropyLossWithIgnore(**kwargs)

    def forward(self, pred, target, **kwargs):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target, **kwargs)
