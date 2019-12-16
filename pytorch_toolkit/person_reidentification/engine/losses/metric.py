import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CenterLoss(nn.Module):
    """Implementation of the Center loss from https://ydwen.github.io/papers/WenECCV16.pdf"""

    def __init__(self, num_classes, embed_size, cos_dist=True):
        super().__init__()
        self.cos_dist = cos_dist
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, embed_size).cuda())
        self.embed_size = embed_size
        self.mse = nn.MSELoss(reduction='elementwise_mean')

    def get_centers(self):
        """Returns estimated centers"""
        return self.centers

    def forward(self, features, labels):
        features = F.normalize(features)
        batch_size = labels.size(0)
        features_dim = features.size(1)
        assert features_dim == self.embed_size

        if self.cos_dist:
            self.centers.data = F.normalize(self.centers.data, p=2, dim=1)

        centers_batch = self.centers[labels, :]

        if self.cos_dist:
            cos_sim = nn.CosineSimilarity()
            cos_diff = 1. - cos_sim(features, centers_batch)
            center_loss = torch.sum(cos_diff) / batch_size
        else:
            center_loss = self.mse(centers_batch, features)

        return center_loss


class GlobalPushPlus(nn.Module):
    """Implementation of the Global Push Plus loss from https://arxiv.org/abs/1812.02465"""

    def __init__(self):
        super().__init__()

    def forward(self, features, centers, labels):
        features = F.normalize(features)
        loss_value = 0
        batch_centers = centers[labels, :]
        labels = labels.cpu().data.numpy()
        assert len(labels.shape) == 1

        center_ids = np.arange(centers.shape[0], dtype=np.int32)
        different_class_pairs = labels.reshape([-1, 1]) != center_ids.reshape([1, -1])

        pos_distances = 1.0 - torch.sum(features * batch_centers, dim=1)
        neg_distances = 1.0 - torch.mm(features, torch.t(centers))

        losses = F.softplus(pos_distances.view(-1, 1) - neg_distances)

        valid_pairs = (different_class_pairs * (losses.cpu().data.numpy() > 0.0)).astype(np.float32)
        num_valid = float(np.sum(valid_pairs))

        if num_valid > 0:
            loss_value = torch.sum(losses * torch.from_numpy(valid_pairs).cuda())
        else:
            return loss_value

        return loss_value / num_valid


class MetricLosses:
    """Class-aggregator for metric-learning losses"""

    def __init__(self, classes_num, embed_size, writer, loss_balancing=False,
                 center_coeff=0.0, glob_push_plus_loss_coeff=0.0,
                 centers_lr=0.5, balancing_lr=0.01):
        self.writer = writer
        self.total_losses_num = 0

        self.center_loss = CenterLoss(classes_num, embed_size, cos_dist=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=centers_lr)
        assert center_coeff >= 0
        self.center_coeff = center_coeff
        if self.center_coeff > 0:
            self.total_losses_num += 1

        self.glob_push_plus_loss = GlobalPushPlus()
        assert glob_push_plus_loss_coeff >= 0
        self.glob_push_plus_loss_coeff = glob_push_plus_loss_coeff
        if self.glob_push_plus_loss_coeff > 0:
            self.total_losses_num += 1

        self.loss_balancing = loss_balancing
        if self.loss_balancing and self.total_losses_num > 1:
            self.loss_weights = nn.Parameter(torch.FloatTensor(self.total_losses_num).cuda())
            self.balancing_optimizer = torch.optim.SGD([self.loss_weights], lr=balancing_lr)
            for i in range(self.total_losses_num):
                self.loss_weights.data[i] = 0.

    def _balance_losses(self, losses):
        assert len(losses) == self.total_losses_num
        for i, loss_val in enumerate(losses):
            losses[i] = torch.exp(-self.loss_weights[i]) * loss_val + \
                            0.5 * self.loss_weights[i]
        return sum(losses)

    def __call__(self, features, labels, epoch_num, iteration):
        all_loss_values = []
        center_loss_val = 0
        if self.center_coeff > 0.:
            center_loss_val = self.center_loss(features, labels)
            all_loss_values.append(center_loss_val)
            self.last_center_val = center_loss_val
            if self.writer is not None:
                self.writer.add_scalar('Loss/center_loss', center_loss_val, iteration)

        glob_push_plus_loss_val = 0
        if self.glob_push_plus_loss_coeff > 0.0 and self.center_coeff > 0.0:
            glob_push_plus_loss_val = self.glob_push_plus_loss(features, self.center_loss.get_centers(), labels)
            all_loss_values.append(glob_push_plus_loss_val)
            if self.writer is not None:
                self.writer.add_scalar('Loss/global_push_plus_loss', glob_push_plus_loss_val, iteration)

        if self.loss_balancing and self.total_losses_num > 1:
            loss_value = self.center_coeff * self._balance_losses(all_loss_values)
            self.last_loss_value = loss_value
        else:
            loss_value = self.center_coeff * center_loss_val + \
                        + self.glob_push_plus_loss_coeff * glob_push_plus_loss_val

        if self.total_losses_num > 0:
            if self.writer is not None:
                self.writer.add_scalar('Loss/AUX_losses', loss_value, iteration)

        return loss_value

    def init_iteration(self):
        """Initializes a training iteration"""
        if self.center_coeff > 0.:
            self.optimizer_centloss.zero_grad()

        if self.loss_balancing:
            self.balancing_optimizer.zero_grad()

    def end_iteration(self):
        """Finalizes a training iteration"""
        if self.loss_balancing and self.total_losses_num > 1:
            self.last_loss_value.backward(retain_graph=True)
            self.balancing_optimizer.step()

        if self.center_coeff > 0.:
            self.last_center_val.backward(retain_graph=True)
            for param in self.center_loss.parameters():
                param.grad.data *= (1. / self.center_coeff)
            self.optimizer_centloss.step()
