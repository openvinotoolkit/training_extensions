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
    stage1_url = config['stage1']['url']
    stage1_path = config['stage1']['dest_path']
    stage2_url = config['stage2']['url']
    stage2_path = config['stage2']['dest_path']
    download_and_extract(path=stage1_path, url=stage1_url, expath='model_weights/')
    download_and_extract(path=stage2_path, url=stage2_url, expath='model_weights/')

def download_data():
    config = get_config(action = 'download')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        os.makedirs(os.path.join('test_data','prepared'))
        os.makedirs(os.path.join('test_data','prepared','bags'))
    data_url = config['test_data']['url']
    data_path = config['test_data']['dest_path']
    download_and_extract(path=data_path, url=data_url, expath='test_data/rbis_ddsm_sample/')
