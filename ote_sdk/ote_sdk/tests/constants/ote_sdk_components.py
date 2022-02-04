"""
OTE SDK components markers.
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


try:
    from e2e.markers.mark_meta import MarkMeta
except ImportError:

    class MarkMeta:  # type: ignore[no-redef]
        """
        Empty marker.
        """


class OteSdkComponent(MarkMeta):
    """
    OTE SDK Component marker.
    """

    OTE_SDK = "ote_sdk"
