from .get_config import get_config
import os
import zipfile
import gdown


def download_and_extract(path, url, expath):
    gdown.download(url, path, fuzzy=True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)


def download_checkpoint(phase):
    config = get_config(action='download', phase=phase, config_path='configs/')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    stage1_url = config['stage1']['drive_url_model']
    stage1_path = config['stage1']['dest_path_model']
    stage2_url = config['stage2']['drive_url_model']
    stage2_path = config['stage2']['dest_path_model']
    download_and_extract(path=stage1_path, url=stage1_url,
                         expath='model_weights/')
    download_and_extract(path=stage2_path, url=stage2_url,
                         expath='model_weights/')


def download_data(phase):
    config = get_config(action='download', phase = phase, config_path='configs/')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        os.makedirs(os.path.join('test_data', 'phase1data'))
        os.makedirs(os.path.join('test_data', 'phase1data', 'train'))
        os.makedirs(os.path.join('test_data', 'phase1data', 'test'))
        
        data_url = config['stage1']['drive_url_data']
        data_path = config['stage1']['dest_path_data']
        download_and_extract(path=data_path, url=data_url, expath='test_data/phase1data/train')

        data_url = config['test_data']['drive_url']
        data_path = config['test_data']['dest_path']
        download_and_extract(path=data_path, url=data_url, expath='test_data/phase1data/test')

        os.makedirs(os.path.join('test_data', 'phase2data'))
        os.makedirs(os.path.join('test_data', 'phase2data', 'train'))
        os.makedirs(os.path.join('test_data', 'phase2data', 'train', 'latent'))
        os.makedirs(os.path.join('test_data', 'phase2data', 'train', 'gd_truth'))
                    
        os.makedirs(os.path.join('test_data', 'phase2data', 'test'))
        os.makedirs(os.path.join('test_data', 'phase2data', 'test', 'latent'))
        os.makedirs(os.path.join('test_data', 'phase2data', 'test', 'gd_truth'))

        data_url = config['stage2']['drive_url_data']
        data_path = config['stage2']['dest_path_data']
        download_and_extract(path=data_path, url=data_url, expath='test_data/phase2data/train')
        