# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import cast

from behave.runner import Context
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from .base import BaseServerRunner


class DockerRunner(BaseServerRunner):
    def __init__(self, context: Context):
        super().__init__(context)
        self.container = None

    def start_server(self):
        image_tag = os.getenv("IMAGE_TAG")
        if not image_tag:
            raise RuntimeError("Environment variable IMAGE_TAG not set")

        tmp_dir = cast(Path, self.tmp_dir)
        port = 7860
        self.container = (
            DockerContainer(image=image_tag)
            .with_env("DATA_DIR", "/application/backend/data")
            .with_env("LOG_DIR", "/application/backend/logs")
            .with_env("HOME", "/tmp")
            .with_env("USER", "runner")
            .with_volume_mapping(str(tmp_dir / "data"), "/application/backend/data", mode="rw")
            .with_volume_mapping(str(tmp_dir / "logs"), "/application/backend/logs", mode="rw")
            .with_bind_ports(port, port)
            .with_kwargs(
                user=f"{os.getuid()}:{os.getgid()}",
                shm_size="2g",
            )
        )
        self.container.start()
        wait_for_logs(self.container, "Application startup completed", timeout=60)

        host = self.container.get_container_host_ip()
        self.base_url = f"https://{host}:{port}"
        self.context.base_url = self.base_url

    def stop_server(self):
        if self.container:
            self.container.stop()
