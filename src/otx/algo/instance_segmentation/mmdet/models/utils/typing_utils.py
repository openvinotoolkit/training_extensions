# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""
from __future__ import annotations

from typing import TypeAlias

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

# Type hint of config data
ConfigType: TypeAlias = ConfigDict | dict
OptConfigType: TypeAlias = ConfigType | None

# Type hint of one or more config data
MultiConfig: TypeAlias = ConfigType | list[ConfigType]
OptMultiConfig: TypeAlias = MultiConfig | None

InstanceList: TypeAlias = list[InstanceData]
OptInstanceList: TypeAlias = InstanceList | None
