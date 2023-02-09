from .get_config import get_config
import os
import zipfile
import wget

def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint(gnn=True):
    config = get_config(action='download', gnn=gnn, config_path='configs/')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    url1 = config['fl_with_gnn']['url_model']
    path1 = config['fl_with_gnn']['dest_path_model']
    url2 = config['fl_without_gnn']['url_model']
    path2 = config['fl_without_gnn']['dest_path_model']
    download_and_extract(path=path1, url=url1,
                         expath='model_weights/')
    download_and_extract(path=path2, url=url2,
                         expath='model_weights/')

def download_data(gnn=True):
    config = get_config(action='download', gnn = gnn, config_path='configs/')
    if not os.path.exists('data'):
        os.makedirs('data')
    url1 = config['data']['url_data']
    path1 = config['data']['dest_path_data']
    url2 = config['data']['url_split']
    path2 = config['data']['dest_path_split']
    download_and_extract(path=path1, url=url1, expath='data/')
    download_and_extract(path=path2, url=url2, expath='data/')
