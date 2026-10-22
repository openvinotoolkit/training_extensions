# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from app.core.logging import extract_log_text
from app.core.logging.utils import MALFORMED_LOG_LINE


class TestExtractLogText:
    @pytest.mark.parametrize(
        "line, expected",
        [
            (json.dumps({"text": "Training started\n", "record": {}}) + "\n", "Training started\n"),
            (json.dumps({"record": {}}) + "\n", ""),
            (json.dumps({"text": None}) + "\n", ""),
            ("\n", ""),
            ("", ""),
            ("not a json line\n", MALFORMED_LOG_LINE),
            ("[1, 2, 3]\n", MALFORMED_LOG_LINE),
        ],
        ids=["text", "missing-text", "non-string-text", "blank-line", "empty", "invalid-json", "non-object-json"],
    )
    def test_extract_log_text(self, line: str, expected: str) -> None:
        assert extract_log_text(line) == expected
