import os
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_checkpoint():
    os.makedirs('model_weights')
    gdd.download_file_from_google_drive(file_id='13KtMdL-RJoTXGRZ8WIdcHkf0XuX9bMMN',
                                    dest_path='model_weights/chest_xray_screening.pth',
                                    unzip=False)
    gdd.download_file_from_google_drive(file_id='1HUmG-wKRoKYxBdwu0_LX1ascBRmA-z5e',
                                    dest_path='model_weights/chest_xray_screening_eff.pth',
                                    unzip=False)