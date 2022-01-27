"""
OTE CLI components markers.
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


class OteCliComponent(MarkMeta):
    """
    OTE CLI Component marker.
    """

    OTE_CLI = "ote_cli"
