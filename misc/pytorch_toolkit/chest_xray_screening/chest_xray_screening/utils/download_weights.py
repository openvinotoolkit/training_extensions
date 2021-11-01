from .get_config import get_config
import os
import zipfile

def download_checkpoint():
    config = get_config(action = 'download')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    densenet_url = config['densenet']['url']
    densenet_path = config['densenet']['dest_path']
    denseneteff_url = config['densenet_eff']['url']
    denseneteff_path = config['densenet_eff']['dest_path']
    os.system(f'wget -mnv {densenet_url} -O {densenet_path}')
    with zipfile.ZipFile(config['densenet']['dest_path'], 'r') as zip_ref:
        zip_ref.extractall('model_weights/')
    os.system(f'wget -mnv {denseneteff_url} -O {denseneteff_path}')
    with zipfile.ZipFile(config['densenet_eff']['dest_path'], 'r') as zip_ref2:
        zip_ref2.extractall('model_weights/')


def download_data():
    config = get_config(action = 'download')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    data_url = config['test_data']['url']
    data_path = config['test_data']['dest_path']
    os.system(f'wget -mnv {data_url} -O {data_path}')
    with zipfile.ZipFile(config['test_data']['dest_path'], 'r') as zip_ref:
        zip_ref.extractall('test_data/chest_xray_data/')
