from .read_config import load_config
import zipfile
import wget


def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)

def download_data():
    data_config = "configs/download_config.json"
    config = load_config(data_config)
    download_and_extract(path=config['toy_dataset_model_save_path'], url=config['toy_dataset_model_url'], expath='')


def main():
    download_data()


if __name__ == '__main__':
    main()
