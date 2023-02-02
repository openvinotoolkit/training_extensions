from src.utils.get_config import get_config
import os
import zipfile
import wget

def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../../")
    config = get_config(action = 'download')
    model_dir = os.path.join(parent_dir,'src','utils','model_weights')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    encoder_url = config['encoder']['url']
    encoder_path = config['encoder']['dest_path']
    download_and_extract(path=parent_dir + encoder_path, url=encoder_url, expath=model_dir)

def download_data():
    config = get_config(action = 'download')
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../../")
    ex_path = parent_dir + '/src/utils/'
    data_url = config['data']['url']
    data_path = parent_dir + config['data']['dest_path']
    download_and_extract(path=data_path, url=data_url, expath=ex_path)
    