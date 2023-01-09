from src.utils.get_config import get_config
import os
import zipfile
# import subprocess
import wget

def download_and_extract(path, url, expath):
    # export_command = f'curl {url} --output {path}'
    # subprocess.run(export_command, shell = True, check = False)
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../../")
    config = get_config(action = 'download')
    model_dir = os.path.join(parent_dir,'src','utils','model_weights')
    # model_dir = parent_dir + '/src/utils/model_weights/'
    # print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    encoder_url = config['encoder']['url']
    encoder_path = config['encoder']['dest_path']
    download_and_extract(path=parent_dir + encoder_path, url=encoder_url, expath=model_dir)

def download_data():
    config = get_config(action = 'download')
    # if not os.path.exists('test_data'):
    #     os.makedirs('test_data')
    current_dir =  os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.abspath(current_dir + "/../../")
    ex_path = parent_dir + '/src/utils/'
    #print(ex_path)
    data_url = config['data']['url']
    data_path = parent_dir + config['data']['dest_path']
    #(data_path)
    download_and_extract(path=data_path, url=data_url, expath=ex_path)
    