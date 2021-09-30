from google_drive_downloader import GoogleDriveDownloader as gdd
import os
import json

def download_checkpoint():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    config_path = os.path.join(root_path, 'configs')
    with open(os.path.join(config_path, 'gdrive_configs.json')) as f1:
            config = json.load(f1)
    gdd.download_file_from_google_drive(file_id = config['distill']['file_id'],
                                    dest_path = config['distill']['dest_path'],
                                    unzip=False)

