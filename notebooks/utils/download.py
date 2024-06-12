"""OTX Notebooks - Utilitiy Download Functions."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pathlib
<<<<<<< HEAD
=======
<<<<<<< HEAD
import zipfile

import requests

=======
import requests
>>>>>>> fdd5dec09f027c1b04f134a1a59eea59455b77bc
import zipfile
>>>>>>> 9bf7b66184e1f751cc28c76a369d91b072e22ba6

import requests


def download_dataset(url: str, extract_to: str) -> None:
    """Downloads and extracts a dataset from a given URL.

    Args:
        url (str): The URL of the dataset zip file.
        extract_to (str): The directory where the dataset should be extracted.

    Example:
        >>> url = "https://github.com/openvinotoolkit/training_extensions/releases/download/"
        ...        "fruits_and_vegetables_dataset/fruits_and_vegetables.zip"
        >>> extract_to = 'data/fruits_and_vegetables/'
        >>> setup_dataset(url, extract_to)
    """
    # Create the directory using pathlib
    dataset_dir = pathlib.Path(extract_to)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check if the dataset has already been extracted
    if not any(dataset_dir.iterdir()):
        # Define the local file path for the zip file
        zip_path = dataset_dir / 'dataset.zip'

        # Download the dataset
        response = requests.get(url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)

        # Unzip the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Remove the zip file
        zip_path.unlink()
    else:
        print(f"The dataset is already available in {dataset_dir}")
