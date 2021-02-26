from torch.nn import functional as F
from torch import nn

import torch

from .model import create_model
from .utils import load_state


class LogitKLDivLoss(nn.Module):
    """Kullback–Leibler divergence loss. Inputs predicted and ground truth logits.

    Args:
        T (float): Softmax temperature.
    """

    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def forward(self, p_logits, q_logits, **kwargs):
        log_p = F.log_softmax(p_logits / self.T, dim=1)
        q = F.softmax(q_logits / self.T, dim=1)
        return F.kl_div(log_p, q, reduction='batchmean') * self.T ** 2


class DistillationLoss(nn.Module):
    """Knowledge distillation loss.

    Args:
        teacher_model (torch.nn.Module): Model that will be used for supervision.
        T (float): Softmax temperature.
    """

    def __init__(self, teacher_model, T=1):
        super().__init__()
        self.teacher_model = teacher_model
        self.kl_div = LogitKLDivLoss(T)

    def forward(self, outputs, inputs, **kwargs):
        """
        Args:
            outputs: Predicted student model logits
            inputs: Inputs that have been used to produce outputs.
        """
        with torch.no_grad():
            teacher_logits = self.teacher_model(*inputs)
        return self.kl_div(outputs, teacher_logits)


class SoftmaxLoss(nn.Module):
    """Classification loss"""

    def forward(self, outputs, targets, **kwargs):
        return F.cross_entropy(outputs, targets)


class WeightedSumLoss(nn.Module):
    """Aggregate multiple loss functions in one weighted sum."""

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.values = {}

    def forward(self, outputs, **kwargs):
        total_loss = outputs.new(1).zero_()
        for loss in self.losses:
            loss_val = self.losses[loss](outputs=outputs, **kwargs)
            total_loss += self.weights[loss] * loss_val
            self.values[loss] = loss_val

        if self.normalize:
            total_loss /= sum(self.weights.values())

        return total_loss

    def add_loss(self, name, loss, weight=1.0):
        self.weights[name] = weight
        self.losses.add_module(name, loss)


def create_criterion(args):
    criterion = WeightedSumLoss()
    softmax = SoftmaxLoss()
    criterion.add_loss('softmax', softmax)

    if args.teacher_model:
        teacher_model, _ = create_model(args, args.teacher_model)

        checkpoint = torch.load(str(args.teacher_checkpoint))
        load_state(teacher_model, checkpoint['state_dict'])
        teacher_model.eval()

        distillation_loss = DistillationLoss(teacher_model, T=8)
        criterion.add_loss(distillation_loss, 0.4)

    return criterion
