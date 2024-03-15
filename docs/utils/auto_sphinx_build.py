# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This script enable automatic detect the change of docs/source contents.

It brings the convinient to check the output of the docs.

When you used the VSCode + watchdog, you can easily check the docs output.

1. Open the builded index.rst file with live server (Visual Studio Code extension)
2. Execute this script `auto_sphix_build_for_vscode.py`
3. Then, the watchdog + sphinx builder automatically builds the source when there are changes

"""

import os
import sys
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class SphinxBuilder(FileSystemEventHandler):
    """Build sphinx docs."""

    def on_modified(self, event: Any) -> None:  # noqa:ANN401
        """When the file is modified."""
        print(f"Changes detected: {event.src_path}")
        os.system("sphinx-build -b html ../source ../build")  # noqa:S605, S607


def main(path: str) -> None:
    """Main function."""
    event_handler: FileSystemEventHandler = SphinxBuilder()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    print("Auto sphinx builder is ON. It automatically builds the source when the change is detected.")
    script_location = Path(__file__).resolve().parent
    parent_dir = script_location.parent
    source_folder = parent_dir / "source"
    print(f"The location of source folder: {source_folder}")
    print("Initial build...")
    time.sleep(10)
    os.system("sphinx-build -b html ../source ../build")  # noqa:S605, S607
    path: str = sys.argv[1] if len(sys.argv) > 1 else str(source_folder)
    main(path)
