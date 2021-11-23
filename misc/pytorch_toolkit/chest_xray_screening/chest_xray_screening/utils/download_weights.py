from .get_config import get_config
import os
import zipfile
import pycurl

def download_checkpoint():
    config = get_config(action = 'download')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    densenet_url = config['densenet']['url']
    densenet_path = config['densenet']['dest_path']
    denseneteff_url = config['densenet_eff']['url']
    denseneteff_path = config['densenet_eff']['dest_path']

    with open(densenet_path, 'wb') as f:
        cl = pycurl.Curl()
        cl.setopt(cl.URL, densenet_url)
        cl.setopt(cl.WRITEDATA, f)
        cl.perform()
        cl.close()
    with zipfile.ZipFile(config['densenet']['dest_path'], 'r') as zip_ref:
        zip_ref.extractall('model_weights/')

    with open(denseneteff_path, 'wb') as f:
        cl = pycurl.Curl()
        cl.setopt(cl.URL, denseneteff_url)
        cl.setopt(cl.WRITEDATA, f)
        cl.perform()
        cl.close()
    with zipfile.ZipFile(config['densenet_eff']['dest_path'], 'r') as zip_ref2:
        zip_ref2.extractall('model_weights/')


def download_data():
    config = get_config(action = 'download')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    data_url = config['test_data']['url']
    data_path = config['test_data']['dest_path']
    with open(data_path, 'wb') as f:
        cl = pycurl.Curl()
        cl.setopt(cl.URL, data_url)
        cl.setopt(cl.WRITEDATA, f)
        cl.perform()
        cl.close()
    with zipfile.ZipFile(config['test_data']['dest_path'], 'r') as zip_ref:
        zip_ref.extractall('test_data/chest_xray_data/')
