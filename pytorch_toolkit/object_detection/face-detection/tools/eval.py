import argparse
import yaml
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


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap


def get_sha256(path, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    os.system(f'sha256sum {path} > {work_dir}/sha256.txt')
    with open(f'{work_dir}/sha256.txt') as f:
        sha256 = f.readlines()[0].strip().split(' ')[0]
    return sha256


def get_size(path, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    os.system(f'ls -l {path} > {work_dir}/ls.txt')
    with open(f'{work_dir}/ls.txt') as f:
        size = f.readlines()[0].strip().split(' ')[4]
    return int(size)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('config',
                      help='A path to model training configuration file (.py).')
    args.add_argument('snapshot',
                      help='A path to pre-trained snapshot (.pth).')
    args.add_argument('out',
                      help='A path to output file where models metrics will be saved (.yml).')
    args.add_argument('--wider_dir',
                      help='Specify this  path if you would like to test your model on WiderFace dataset.')

    return args.parse_args()


def compute_wider_metrics(face_detection_tools, config_path, res_pkl, work_dir, wider_dir, outputs):
    wider_data_folder = wider_dir
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
        f' {config_path} {res_pkl} {wider_face_predictions}'.split(' '))
    print(wider_face_predictions)
    res_wider_metrics = os.path.join(work_dir, "wider_metrics.json")
    subprocess.run(
        f'python {face_detection_tools}/wider_face_eval.py'
        f' -g {wider_data_folder}/eval_tools/ground_truth/'
        f' -p {wider_face_predictions}'
        f' --out {res_wider_metrics}'.split(' '))
    with open(res_wider_metrics) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


def coco_ap_eval(mmdetection_tools, config_path, work_dir, snapshot, res_pkl, outputs):
    with open(os.path.join(work_dir, 'test_py_stdout'), 'w') as test_py_stdout:
        subprocess.run(
            f'python {mmdetection_tools}/test.py'
            f' {config_path} {snapshot}'
            f' --out {res_pkl} --eval bbox'.split(' '), stdout=test_py_stdout)
    ap = collect_ap(os.path.join(work_dir, 'test_py_stdout'))[0]
    outputs.append({'key': 'ap', 'value': ap * 100, 'unit': '%', 'displayName': 'AP @ [IoU=0.50:0.95]'})
    return outputs


def custom_ap_eval(face_detection_tools, config_path, work_dir, res_pkl, outputs):
    res_custom_metrics = os.path.join(work_dir, "custom_metrics.json")
    subprocess.run(
        f'python {face_detection_tools}/wider_custom_eval.py'
        f' {config_path} {res_pkl} --out {res_custom_metrics}'.split(' '))
    with open(res_custom_metrics) as read_file:
        ap_64x64 = [x['average_precision'] for x in json.load(read_file) if x['object_size'][0] == 64][0]
        outputs.append({'key': 'ap_64x64', 'value': ap_64x64, 'displayName': 'AP for faces > 64x64'})
    return outputs


def get_complexity_and_size(mmdetection_tools, cfg, config_path, work_dir, outputs):
    image_shape = [x['img_scale'] for x in cfg.test_pipeline if 'img_scale' in x][0][::-1]
    image_shape = " ".join([str(x) for x in image_shape])

    res_complexity = os.path.join(work_dir, "complexity.json")

    subprocess.run(
        f'python {mmdetection_tools}/get_flops.py'
        f' {config_path}'
        f' --shape {image_shape}'
        f' --out {res_complexity}'.split(' '))
    with open(res_complexity) as read_file:
        content = json.load(read_file)
        outputs.extend(content)
    return outputs


def get_file_size_and_sha256(snapshot, work_dir):
    return {
        'sha256': get_sha256(snapshot, work_dir),
        'size': get_size(snapshot, work_dir),
        'name': os.path.basename(snapshot),
        'source': snapshot
    }


def eval(config_path, snapshot, wider_dir, out):
    install_and_import('gdown')
    mmdetection_tools = '../../external/mmdetection/tools'
    face_detection_tools = 'face-detection/tools'

    cfg = Config.fromfile(config_path)

    work_dir = tempfile.mkdtemp()
    print('results are stored in:', work_dir)

    files = get_file_size_and_sha256(snapshot, work_dir)

    metrics = []
    res_pkl = os.path.join(work_dir, "res.pkl")
    metrics = coco_ap_eval(mmdetection_tools, config_path, work_dir, snapshot, res_pkl, metrics)
    metrics = custom_ap_eval(face_detection_tools, config_path, work_dir, res_pkl, metrics)

    if wider_dir:
        metrics = compute_wider_metrics(face_detection_tools, config_path, res_pkl, work_dir, wider_dir, metrics)

    metrics = get_complexity_and_size(mmdetection_tools, cfg, config_path, work_dir, metrics)

    for metric in metrics:
        metric['value'] = round(metric['value'], 3)

    outputs = {
        'files': [files],
        'metrics': metrics
    }

    if os.path.exists(out):
        with open(out) as read_file:
            content = yaml.load(read_file)
        content.update(outputs)
        outputs = content

    with open(out, 'w') as write_file:
        yaml.dump(outputs, write_file)


def main():
    args = parse_args()
    eval(args.config, args.snapshot, args.wider_dir, args.out)


if __name__ == '__main__':
    main()
