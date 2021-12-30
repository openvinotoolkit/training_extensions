import os
from .download_weights import download_checkpoint, download_data

def make_dirs():
    os.mkdir('downloads')
    os.mkdir('downloads/data/')
    download_data()
    os.mkdir('downloads/checkpoints/')
    download_checkpoint()
    os.mkdir('temp_data')
    os.mkdir('temp_data/infer_results/')
    os.mkdir('temp_data/checkpoints/')
    # os.mkdir('temp_data/infer_results/model_1')

