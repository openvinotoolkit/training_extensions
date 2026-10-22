# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from app.core.logging import iter_log_text
from app.core.logging.utils import MALFORMED_LOG_LINE


def test_iter_log_text(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "Training started\n", "record": {}}),
                json.dumps({"record": {}}),
                "not a json line",
                "[1, 2, 3]",
                json.dumps({"text": "Epoch 1/10\n", "record": {}}),
            ]
        )
        + "\n"
    )

    assert list(iter_log_text(log_path)) == [
        "Training started\n",
        MALFORMED_LOG_LINE,
        MALFORMED_LOG_LINE,
        "Epoch 1/10\n",
    ]
