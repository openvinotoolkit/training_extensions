"""OTX Notebooks - Utilitiy IO Functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from urllib.parse import urlparse

from git.repo import Repo


def get_repo_name_from_url(git_url: str) -> str:
    """Extracts the repository name from the Git URL.

    The repository name is derived from the last part of the URL path,
    typically the name of the directory created when the repository is cloned.

    Args:
        git_url (str): The URL of the Git repository.

    Returns:
        str: The name of the repository.

    Example:
        >>> get_repo_name_from_url("https://github.com/openvinotoolkit/training_extensions.git")
        'training_extensions'

        >>> get_repo_name_from_url("https://github.com/user/repo.git")
        'repo'
    """
    parsed_url = urlparse(git_url)
    return Path(parsed_url.path).stem


def setup_repo(git_url: str, branch: str = "main") -> None:
    """Sets up the Git repository by cloning it if necessary and checking out the specified branch.

    This function handles the current directory being in different locations,
    clones the repository if not already present, and checks out the specified branch.

    Args:
        git_url (str): The URL of the Git repository.
        branch (str): The branch to check out. Defaults to 'main'.

    Returns:
        None

    Example:
        >>> setup_repo("https://github.com/openvinotoolkit/training_extensions.git", branch='tutorials/cvpr24')
    """
    current_directory = Path.cwd()
    repo_name = get_repo_name_from_url(git_url)

    if current_directory.name == "notebooks":
        # On the assumption that the notebook is located in ~/repo_name/notebooks/
        root_directory = current_directory.parent
    elif current_directory.name == repo_name:
        # This means that the notebook is run from the main repository directory.
        root_directory = current_directory
    else:
        # Otherwise, we'll need to clone the repository to the `current_directory`
        repo = Repo.clone_from(
            url=git_url,
            to_path=current_directory,
        )
        root_directory = current_directory / repo_name
        # Check out the specified branch
        repo.git.checkout(branch)

    os.chdir(root_directory)

    # If you need to check out the branch even when the repository is already cloned
    if root_directory.exists() and (root_directory / ".git").exists():
        repo = Repo(root_directory)
        repo.git.checkout(branch)


def seed_everywhere(seed: int = 42) -> None:
    """Sets the seed for random number generators in np & torch libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.

    Example:
        >>> seed_everywhere(seed=42)
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # os
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
