"""OTX Workspace Module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import OmegaConf

from otx.v2.api.utils.importing import get_otx_root_path


def set_workspace(root: Optional[str] = None, name: str = "otx-workspace") -> str:
    """Set workspace path according to arguments.

    Args:
    ----
        root (str, optional): Target root path. Defaults to None.
        name (str, optional): Name of workspace folder. Defaults to "otx-workspace".

    Returns:
    -------
        str: Workspace folder path.
    """
    return f"{root}/{name}" if root else f"./{name}"


class Workspace:
    """A class representing an OTX workspace."""

    def __init__(self, work_dir: Optional[str] = None, task: Optional[str] = None) -> None:
        """Initialize a Workspace object.

        Args:
        ----
            work_dir (Optional[str]): The path to the workspace directory.
                If None, a new workspace directory will be created.
            task (Optional[str]): The name of the task associated with this workspace.

        Returns:
        -------
            None
        """
        self.otx_root = get_otx_root_path()
        self.work_dir = Path(work_dir) if work_dir is not None else Path(set_workspace()).resolve()
        self.task = task
        self.mkdir_or_exist()
        self.latest_dir = self.work_dir / "latest"
        self._config: Dict[str, Any] = {}
        self._config_path = self.work_dir / "configs.yaml"

    @property
    def config_path(self) -> Path:
        """Returns the path to the configuration file for the workspace."""
        return self._config_path

    def check_workspace(self) -> bool:
        """Check that the class's work_dir is an actual workspace folder.

        Returns:
        -------
            bool: true for workspace else false
        """
        return (self.work_dir / "configs.yaml").exists()

    def mkdir_or_exist(self) -> None:
        """If the workspace doesn't exist, create it."""
        if self.task is not None:
            self.work_dir = self.work_dir / self.task
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f"[*] Workspace Path: {self.work_dir}")

    def dump_config(
        self,
        config: Optional[Union[str, Path, Dict]] = None,
        filename: Optional[Union[str, Path]] = None,
    ) -> None:
        """Dump output configuration.

        Args:
        ----
            config (Optional[Union[str, Path, Dict]], optional): Config contents to be save. Defaults to None.
            filename (Optional[Union[str, Path]], optional): Output config file path. Defaults to None.

        Raises:
        ------
            FileNotFoundError: If config is a string or Path and it doesn't exist, raise an error.
        """
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

    def add_config(self, config: dict) -> None:
        """Update workspace's configuration.

        Args:
        ----
            config: Config contents.
        """
        self._config.update(config)

    def update_latest(self, target_dir: Path) -> None:
        """Update the latest folder symbolic link to models.

        If the latest directory already exists, it is unlinked. If the parent directory of the latest directory does not
        exist, it is created. Finally, the latest directory is symlinked to the target directory.

        Args:
        ----
            target_dir (Path): The target directory to symlink to.

        Returns:
        -------
            None
        """
        if self.latest_dir.exists():
            self.latest_dir.unlink()
        elif not self.latest_dir.parent.exists():
            self.latest_dir.parent.mkdir(exist_ok=True, parents=True)
        self.latest_dir.symlink_to(target_dir.resolve())
