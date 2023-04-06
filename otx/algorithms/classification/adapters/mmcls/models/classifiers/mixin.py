"""Module defining Mix-in class of SAMClassifier."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class SAMClassifierMixin:
    """SAM-enabled BaseClassifier mix-in."""

    def train_step(self, data, optimizer=None, **kwargs):
        """Saving current batch data to compute SAM gradient."""
        self.current_batch = data
        return super().train_step(data, optimizer, **kwargs)


class LossDynamicsTrackingMixin:
    """Mix-in to track loss dynamics during training."""

    def __init__(self, track_loss_dynamics: bool = False, **kwargs):
        if getattr(self, "multilabel") or getattr(self, "hierarchical"):
            raise RuntimeError("multilabel or hierarchical tasks are not supported now.")

        if track_loss_dynamics:
            head_cfg = kwargs.get("head", None)
            loss_cfg = head_cfg.get("loss", None)
            loss_cfg["reduction"] = "none"

        self._track_loss_dynamics = track_loss_dynamics
        super().__init__(**kwargs)

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step for training.

        If self._track_loss_dynamics = False, just follow BaseClassifier.train_step().
        Otherwise, it steps with tracking loss dynamics.
        """
        if self._track_loss_dynamics:
            return self._train_step_with_tracking(data, optimizer, **kwargs)
        return super().train_step(data, optimizer, **kwargs)

    def _train_step_with_tracking(self, data, optimizer=None, **kwargs):
        losses = self(**data)
        loss_dyns = losses["loss"].detach().cpu().numpy()
        gt_labels = data["gt_label"].detach().cpu().numpy()
        entity_ids = [img_meta["entity_id"] for img_meta in data["img_metas"]]
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            loss_dyns=loss_dyns,
            entity_ids=entity_ids,
            gt_labels=gt_labels,
            num_samples=len(data["img"].data),
        )

        return outputs
