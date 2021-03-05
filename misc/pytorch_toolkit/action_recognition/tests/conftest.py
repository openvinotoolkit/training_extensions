from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent


@pytest.fixture
def data_path():
    return TEST_DIR / 'test_data'


@pytest.fixture
def rand():
    return np.random.RandomState(seed=1)
