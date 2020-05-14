import argparse
import json
import subprocess
import os
import tempfile

from mmcv.utils import Config


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.run('python -m pip install gdown'.split(' '))
    finally:
        globals()[package] = importlib.import_module(package)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('config')
    args.add_argument('gpu_num')
    args.add_argument('--out')
    args.add_argument('--compute_wider_metrics', action='store_true')
    args.add_argument('--wider_dir', default='wider')

    return args.parse_args()


def main():
    install_and_import('gdown')
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir

    mmdetection_tools = '../../external/mmdetection/tools'
    face_detection_tools = 'face-detection/tools'

    subprocess.run(f'{mmdetection_tools}/dist_train.sh'
                   f' {args.config}'
                   f' {args.gpu_num}'.split(' '))
    outputs = []

    snapshot = os.path.join(cfg.work_dir, "latest.pth")
    res_pkl = os.path.join(cfg.work_dir, "res.pkl")
    subprocess.run(
        f'python {mmdetection_tools}/test.py'
        f' {args.config} {snapshot}'
        f' --out {res_pkl}'.split(' '))
    res_custom_metrics = os.path.join(cfg.work_dir, "custom_metrics.json")
    subprocess.run(
        f'python {face_detection_tools}/wider_custom_eval.py'
        f' {args.config} {res_pkl} --out {res_custom_metrics}'.split(' '))
    with open(res_custom_metrics) as read_file:
        ap_64x64 = [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
        outputs.append({'key': 'ap_64x64', 'value': ap_64x64, 'displayName': 'AP for faces > 64x64'})

    if args.compute_wider_metrics:
        wider_data_folder = args.wider_dir
        os.makedirs(wider_data_folder, exist_ok=True)

        wider_data_zip = os.path.join(wider_data_folder, 'WIDER_val.zip')
        if not os.path.exists(wider_data_zip):
            subprocess.run(
                f'gdown https://drive.google.com/uc?id=0B6eKvaijfFUDd3dIRmpvSk8tLUk'
                f' -O {wider_data_zip}'.split(' '))
        if not os.path.exists(os.path.join(wider_data_folder, 'WIDER_val')):
            subprocess.run(f'unzip {wider_data_zip} -d {wider_data_folder}'.split(' '))

        eval_tools_zip = os.path.join(wider_data_folder, 'eval_tools.zip')
        if not os.path.exists(eval_tools_zip):
            subprocess.run(
                f'wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip'
                f' -O {eval_tools_zip}'.split(' '))
        if not os.path.exists(os.path.join(wider_data_folder, 'eval_tools')):
            subprocess.run(f'unzip {eval_tools_zip} -d {wider_data_folder}'.split(' '))

        wider_face_predictions = tempfile.mkdtemp()
        subprocess.run(
            f'python {face_detection_tools}/test_out_to_wider_predictions.py'
            f' {args.config} {res_pkl} {wider_face_predictions}'.split(' '))
        print(wider_face_predictions)
        res_wider_metrics = os.path.join(cfg.work_dir, "wider_metrics.json")
        subprocess.run(
            f'python {face_detection_tools}/wider_face_eval.py'
            f' -g {wider_data_folder}/eval_tools/ground_truth/'
            f' -p {wider_face_predictions}'
            f' --out {res_wider_metrics}'.split(' '))
        with open(res_wider_metrics) as read_file:
            content = json.load(read_file)
            outputs.extend(content)

    image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
    image_shape = " ".join([str(x) for x in image_shape])

    res_complexity = os.path.join(cfg.work_dir, "complexity.json")

    subprocess.run(
        f'python {mmdetection_tools}/get_flops.py'
        f' {args.config}'
        f' --shape {image_shape}'
        f' --out {res_complexity}'.split(' '))
    with open(res_complexity) as read_file:
        content = json.load(read_file)
        outputs.extend(content)

    if args.out:
        with open(args.out, 'w') as write_file:
            json.dump(outputs, write_file, indent=4)


if __name__ == '__main__':
    main()
