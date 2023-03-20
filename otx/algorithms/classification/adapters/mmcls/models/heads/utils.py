# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT
#

from torch import nn


def generate_aux_mlp(aux_mlp_cfg: dict, in_channels: int):
    out_channels = aux_mlp_cfg["out_channels"]
    if out_channels <= 0:
        raise ValueError(f"out_channels={out_channels} must be a positive integer")
    if "hid_channels" in aux_mlp_cfg and aux_mlp_cfg["hid_channels"] > 0:
        hid_channels = aux_mlp_cfg["hid_channels"]
        mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hid_channels, out_features=out_channels),
        )
    else:
        mlp = nn.Linear(in_features=in_channels, out_features=out_channels)

    return mlp


class EMAMeter:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = self.alpha * self.val + (1 - self.alpha) * val


class LossBalancer:
    def __init__(self, num_losses, weights=None, ema_weight=0.7) -> None:
        self.EPS = 1e-9
        self.avg_estimators = [EMAMeter(ema_weight) for _ in range(num_losses)]

        if weights is not None:
            assert len(weights) == num_losses
            self.final_weights = weights
        else:
            self.final_weights = [1.0] * num_losses

    def balance_losses(self, losses):
        total_loss = 0.0
        for i, loss in enumerate(losses):
            self.avg_estimators[i].update(float(loss))
            total_loss += (
                self.final_weights[i] * loss / (self.avg_estimators[i].val + self.EPS) * self.avg_estimators[0].val
            )

        return total_loss
