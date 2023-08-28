"""OTX Workspace Module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import OmegaConf

from otx.v2.api.utils.importing import get_otx_root_path


def set_workspace(root: str = None, name: str = "otx-workspace"):
    """Set workspace path according to arguments."""
    path = f"{root}/{name}" if root else f"./{name}"
    return path


class Workspace:
    def __init__(self, work_dir: Optional[str] = None) -> None:
        self.otx_root = get_otx_root_path()
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.mkdir_or_exist()
        self._config = {}
        self._config_path = self.work_dir / "configs.yaml"

    @property
    def config_path(self):
        return self._config_path

    def check_workspace(self) -> bool:
        """Check that the class's work_dir is an actual workspace folder.

        Returns:
            bool: true for workspace else false
        """
        return (self.work_dir / "configs.yaml").exists()

    def mkdir_or_exist(self):
        if self.work_dir is None:
            self.work_dir = Path(set_workspace()).resolve()
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f"[*] Workspace Path: {self.work_dir}")

    def dump_config(self, config: Optional[Union[str, Path, Dict]] = None, filename: Optional[Union[str, Path]] = None):
        if config is None:
            config = self._config
        if isinstance(config, (str, Path)):
            if not Path(config).is_file():
                raise FileNotFoundError(config)
            config = OmegaConf.load(str(config))
        if filename is None:
            (self.work_dir / "configs.yaml").write_text(OmegaConf.to_yaml(config))
        else:
            Path(filename).write_text(OmegaConf.to_yaml(config))

    def add_config(self, config: Dict):
        self._config.update(config)
