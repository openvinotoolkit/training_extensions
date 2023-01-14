from .get_config import get_config
import os
import zipfile
import wget

def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    config = get_config(action='download', config_path='configs/')
    if not os.path.exists('downloads/model_weights/'):
        os.makedirs('downloads/model_weights/')
    model_url = config['model_weights']['url_model']
    model_path = config['model_weights']['dest_path_model']

    download_and_extract(path=model_path, url=model_url,
                         expath='downloads/model_weights/')

def download_data():
    config = get_config(action='download', config_path='configs/')
    if not os.path.exists('downloads/test_data/'):
        os.makedirs('downloads/test_data/')
        data_url = config['test_data']['url_data']
        data_path = config['test_data']['dest_path_data']
        download_and_extract(path=data_path, url=data_url, expath='downloads/test_data/')

if __name__ == '__main__':
    download_data()
    download_checkpoint()
