from .get_config import get_config
import os
import zipfile
import subprocess


def download_and_extract(path, url, expath):
    export_command = f'curl {url} --output {path}'
    subprocess.run(export_command, shell = True, check = True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    config = get_config(action = 'download')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    densenet_url = config['densenet']['url']
    densenet_path = config['densenet']['dest_path']
    denseneteff_url = config['densenet_eff']['url']
    denseneteff_path = config['densenet_eff']['dest_path']
    download_and_extract(path=densenet_path, url=densenet_url, expath='model_weights/')
    download_and_extract(path=denseneteff_path, url=denseneteff_url, expath='model_weights/')

def download_data():
    config = get_config(action = 'download')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    data_url = config['test_data']['url']
    data_path = config['test_data']['dest_path']
    download_and_extract(path=data_path, url=data_url, expath='test_data/chest_xray_data/')
