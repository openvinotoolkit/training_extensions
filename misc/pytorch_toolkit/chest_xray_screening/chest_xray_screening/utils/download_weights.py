from google_drive_downloader import GoogleDriveDownloader as gdd
from .get_config import get_config

def download_checkpoint():
    config = get_config(action = 'gdrive')
    gdd.download_file_from_google_drive(file_id=config['densenet']['file_id'],
                                    dest_path=config['densenet']['dest_path'],
                                    unzip=False)
    gdd.download_file_from_google_drive(file_id=config['densenet_eff']['file_id'],
                                    dest_path=config['densenet_eff']['dest_path'],
                                    unzip=False)

def download_data():
    config = get_config(action = 'gdrive')
    gdd.download_file_from_google_drive(file_id=config['test_data']['file_id'],
                                    dest_path=config['test_data']['dest_path'],
                                    unzip=True)
