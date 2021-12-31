from .read_config import load_config
import os
import zipfile
import subprocess


def download_and_extract(path, url, expath):
    export_command = f'curl {url} --output {path}'
    subprocess.run(export_command, shell = True, check = True)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_checkpoint():
    checkpoint_config = "configs/download_config.json"
    config = load_config(checkpoint_config)
    if not os.path.exists('downloads/checkpoints'):
        os.makedirs('downloads/checkpoints')
    model_url = config["model_download_url"]
    model_save_path = config["model_save_path"]
    download_and_extract(path=model_save_path, url=model_url, expath='downloads/checkpoints/')

def download_data():
    data_config = "configs/download_config.json"
    config = load_config(data_config)
    if not os.path.exists('downloads/data'):
        os.makedirs('downloads/data')
    real_data_url = config['real_data_download_url']
    real_data_path = config['real_data_save_path']
    stage0_data_url = config['stage0_data_download_url']
    stage0_data_path = config['stage0_data_save_path']
    download_and_extract(path=real_data_path, url=real_data_url, expath='downloads/data/')
    download_and_extract(path=stage0_data_path, url=stage0_data_url, expath='downloads/data/')


def main():
    download_checkpoint()
    download_data()

if __name__ == '__main__':
    main()
