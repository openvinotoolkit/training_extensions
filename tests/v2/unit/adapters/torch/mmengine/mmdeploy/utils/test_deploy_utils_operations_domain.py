# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.v2.adapters.torch.mmengine.mmdeploy.utils.operations_domain import (
    DOMAIN_CUSTOM_OPS_NAME,
    add_domain,
)


def test_add_domain() -> None:
    assert add_domain("abc") == DOMAIN_CUSTOM_OPS_NAME + "::" + "abc"
