from .get_config import get_config
import os
import zipfile
import wget


def download_and_extract(path, url, expath):
    wget.download(url, path)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(expath)


def download_checkpoint(phase):
    config = get_config(action='download', phase=phase, config_path='configs/')
    if not os.path.exists('model_weights'):
        os.makedirs('model_weights')
    stage1_url = config['stage1']['url_model']
    stage1_path = config['stage1']['dest_path_model']
    stage2_url = config['stage2']['url_model']
    stage2_path = config['stage2']['dest_path_model']
    download_and_extract(path=stage1_path, url=stage1_url,
                         expath='model_weights/')
    download_and_extract(path=stage2_path, url=stage2_url,
                         expath='model_weights/')


def download_data(phase):
    config = get_config(action='download', phase = phase, config_path='configs/')
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
        # os.makedirs(os.path.join('test_data', 'phase1'))
        # os.makedirs(os.path.join('test_data', 'phase1', 'train'))
        # os.makedirs(os.path.join('test_data', 'phase1', 'test'))
        
        data_url = config['stage1']['url_data']
        data_path = config['stage1']['dest_path_data']
        download_and_extract(path=data_path, url=data_url, expath='test_data/')

        # data_url = config['test_data']['url_data']
        # data_path = config['test_data']['dest_path']
        # download_and_extract(path=data_path, url=data_url, expath='test_data/phase1/')

        # os.makedirs(os.path.join('test_data', 'phase2'))
        # os.makedirs(os.path.join('test_data', 'phase2data', 'train'))
        # os.makedirs(os.path.join('test_data', 'phase2data', 'train', 'latent'))
        # os.makedirs(os.path.join('test_data', 'phase2data', 'train', 'gd_truth'))
                    
        # os.makedirs(os.path.join('test_data', 'phase2data', 'test'))
        # os.makedirs(os.path.join('test_data', 'phase2data', 'test', 'latent'))
        # os.makedirs(os.path.join('test_data', 'phase2data', 'test', 'gd_truth'))

        data_url = config['stage2']['url_data']
        data_path = config['stage2']['dest_path_data']
        download_and_extract(path=data_path, url=data_url, expath='test_data/')


def main(args):
    download_checkpoint(args.phase)
    download_data(args.phase)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--phase', type=int, required=True, help='train_phase')


    args = parser.parse_args()
    main(args)