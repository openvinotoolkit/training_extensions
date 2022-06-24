from .get_config import get_config
import os
import zipfile
import subprocess


def download_and_extract(path, url, expath):
    export_command = f'curl {url} --output {path}'
    subprocess.run(export_command, shell=True, check=True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)


def download_checkpoint():
    config = get_config(action='download')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    encoder_url = config['encoder']['url']
    encoder_path = config['encoder']['dest_path']
    decoder_url = config['decoder']['url']
    decoder_path = config['decoder']['dest_path']
    download_and_extract(path=encoder_path,
                         url=encoder_url, expath='model_weights/')  # the path may change later
    download_and_extract(path=decoder_path,
                         url=decoder_url, expath='model_weights/')  # the path may change later


def download_data():
    config = get_config(action='download')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    data_url = config['test_data']['url']
    data_path = config['test_data']['dest_path']
    download_and_extract(path=data_path, url=data_url,
                         expath='test_data/chest_xray_data/')
