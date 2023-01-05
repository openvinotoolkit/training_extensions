import pytest
import os
import copy

from pathlib import Path
from tempfile import TemporaryDirectory

@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(autouse=True)
def set_default_tmp_path(tmp_dir_path):
    env = copy.deepcopy(os.environ)
    env["TMPDIR"] = str(tmp_dir_path)
