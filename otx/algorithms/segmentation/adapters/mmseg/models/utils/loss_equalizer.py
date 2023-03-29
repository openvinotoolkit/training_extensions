"""Loss equalizer."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class LossEqualizer:
    """Loss equalizer."""

    def __init__(self, weights=None, momentum=0.1):
        self.momentum = momentum

        self.trg_ratios = None
        if weights is not None:
            assert isinstance(weights, dict)
            assert len(weights) > 0

            sum_weight = 0.0
            for loss_weight in weights.values():
                assert loss_weight > 0
                sum_weight += float(loss_weight)
            assert sum_weight > 0.0

            self.trg_ratios = {loss_name: float(loss_weight) / sum_weight for loss_name, loss_weight in weights.items()}

        self._smoothed_values = dict()

    def reweight(self, losses):
        """Reweight."""
        assert isinstance(losses, dict)

        if len(losses) == 0:
            return losses

        for loss_name, loss_value in losses.items():
            if loss_name not in self._smoothed_values:
                self._smoothed_values[loss_name] = loss_value.item()
            else:
                smoothed_loss = self._smoothed_values[loss_name]
                self._smoothed_values[loss_name] = (
                    1.0 - self.momentum
                ) * smoothed_loss + self.momentum * loss_value.item()

        if len(self._smoothed_values) == 1:
            return losses

        total_sum = sum(self._smoothed_values.values())
        trg_value_default = total_sum / float(len(self._smoothed_values))

        weighted_losses = dict()
        for loss_name, loss_value in losses.items():
            if self.trg_ratios is not None:
                assert loss_name in self.trg_ratios.keys()
                trg_value = self.trg_ratios[loss_name] * total_sum
            else:
                trg_value = trg_value_default

            loss_weight = trg_value / self._smoothed_values[loss_name]
            weighted_losses[loss_name] = loss_weight * loss_value

        return weighted_losses
