# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from otx.algo.classification.efficientnet_b0 import EfficientNetB0ForMulticlassCls
from otx.engine import Engine
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD


def test_warmup_schedule(
    tmp_path: Path,
) -> None:
    tmp_path_train = tmp_path

    model = EfficientNetB0ForMulticlassCls(num_classes=2)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = CosineAnnealingLR(optimizer, 10)

    engine = Engine(
        data_root="tests/assets/classification_dataset",
        work_dir=tmp_path_train,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    max_epochs = 10
    engine.datamodule.config.train_subset.batch_size = 5

    loggers = [CSVLogger(tmp_path_train)]
    callbacks = [LearningRateMonitor("epoch", log_momentum=True)]
    warmup_steps = 10
    engine.train(max_epochs, warmup_steps=warmup_steps, warmup_by_epochs=False, logger=loggers, callbacks=callbacks)

    csv_file = Path.glob(f"{tmp_path_train}/lightning_logs/*/metrics.csv")[0]

    # Initialize a list to store lr-SGD values
    lr_values = []

    # Read the CSV file and extract lr-SGD values
    with Path.open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lr_sgd = row["lr-SGD"]
            if lr_sgd:
                lr_values.append(float(lr_sgd))

    # lr_values include intial lr value at zero index.
    valid_lr_values = lr_values[1:]

    for epoch in range(len(valid_lr_values)):
        steps = (epoch + 1) * 5
        if steps < warmup_steps:
            assert valid_lr_values[epoch] < valid_lr_values[epoch + 1]
        elif epoch < len(valid_lr_values) - 1:
            assert valid_lr_values[epoch] > valid_lr_values[epoch + 1]
