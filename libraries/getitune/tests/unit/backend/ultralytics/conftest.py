# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Skip all ultralytics tests when the optional ultralytics package is not installed."""

import pytest

pytest.importorskip("ultralytics", reason="ultralytics is not installed")
