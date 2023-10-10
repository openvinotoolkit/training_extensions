"""OpenVINO Training Extensions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# FIXME: Issue with setuptools - https://github.com/Madoshakalaka/pipenv-setup/issues/101
import os

__version__ = "2.0.0"


OTX_LOGO: str = """

 ██████╗  ████████╗ ██╗  ██╗
██╔═══██╗ ╚══██╔══╝ ╚██╗██╔╝
██║   ██║    ██║     ╚███╔╝
██║   ██║    ██║     ██╔██╗
╚██████╔╝    ██║    ██╔╝ ██╗
 ╚═════╝     ╚═╝    ╚═╝  ╚═╝

"""


os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
