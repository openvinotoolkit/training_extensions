import os
import wget
import pyunpack
import pandas as pd
import numpy as np

def download_from_url(url, output_path):
    """Downloads a file froma url."""

    print('Pulling data from {} to {}'.format(url, output_path))
    wget.download(url, output_path)
    print('done')

def recreate_folder(path):
    """Deletes and recreates folder."""

    shutil.rmtree(path)
    os.makedirs(path)

def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            'Error in unzipping process! {} not found.'.format(output_file))

def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.
    Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print('Done.')
