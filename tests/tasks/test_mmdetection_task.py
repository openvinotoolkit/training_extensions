# pylint: disable=redefined-outer-name

import os
import tempfile
from subprocess import run

import pytest
from ote import MMDETECTION_DIR
from ote.interfaces.parameters import BaseTaskParameters
from ote.tasks.mmdetection import Dataset, Task
from ote.tests.utils import collect_ap

snapshots_dir = tempfile.mkdtemp()


@pytest.fixture()
def download_face_detection_0200():
    global snapshots_dir
    snapshot_path = os.path.join(snapshots_dir, 'face-detection-0200.pth')
    if not os.path.exists(snapshot_path):
        url = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/face-detection-0200.pth' # pylint: disable=line-too-long
        run(f'wget -q -O {snapshot_path} {url}', shell=True, check=True)
    return snapshot_path


def test_task_training(download_face_detection_0200):
    env_params = BaseTaskParameters.BaseEnvironmentParameters()
    ote_root = os.path.join(os.path.dirname(__file__), '..', '..')
    env_params.config_path = os.path.join(ote_root, 'models', 'object_detection', 'model_templates',
                                          'face-detection', 'face-detection-0200', 'model.py')
    env_params.snapshot_path = download_face_detection_0200
    env_params.work_dir = tempfile.mkdtemp()
    train_params = BaseTaskParameters.BaseTrainingParameters()
    train_params.max_num_epochs = 1
    task = Task(env_params)

    ann_file = os.path.join(ote_root, 'data', 'airport', 'annotation_faces_train.json')
    data_root = os.path.join(ote_root, 'data', 'airport')

    train_data_source = Dataset(ann_file, data_root)
    val_data_source = Dataset(ann_file, data_root)

    task.train(train_data_source, val_data_source, train_params)
    if not os.path.exists(os.path.join(env_params.work_dir, 'latest.pth')):
        raise RuntimeError()


def test_task_testing(download_face_detection_0200):
    env_params = BaseTaskParameters.BaseEnvironmentParameters()
    ote_root = os.path.join(os.path.dirname(__file__), '..', '..')
    env_params.config_path = os.path.join(ote_root, 'models', 'object_detection', 'model_templates',
                                          'face-detection', 'face-detection-0200', 'model.py')
    env_params.snapshot_path = download_face_detection_0200
    env_params.work_dir = tempfile.mkdtemp()
    eval_params = BaseTaskParameters.BaseEvaluationParameters()
    task = Task(env_params)

    ann_file = os.path.join(ote_root, 'data', 'airport', 'annotation_faces_train.json')
    data_root = os.path.join(ote_root, 'data', 'airport')

    test_data_source = Dataset(ann_file, data_root)

    _, eval_results = task.test(test_data_source, eval_params)
    assert eval_results['bbox_mAP'] == 0.138


def test_task_exporting(download_face_detection_0200):
    env_params = BaseTaskParameters.BaseEnvironmentParameters()
    ote_root = os.path.join(os.path.dirname(__file__), '..', '..')
    env_params.config_path = os.path.join(ote_root, 'models', 'object_detection', 'model_templates',
                                          'face-detection', 'face-detection-0200', 'model.py')
    env_params.snapshot_path = download_face_detection_0200
    env_params.work_dir = tempfile.mkdtemp()
    export_params = BaseTaskParameters.BaseExportParameters()
    export_params.save_model_to = tempfile.mkdtemp()
    task = Task(env_params)

    task.export(export_params)
    assert os.path.exists(os.path.join(export_params.save_model_to, 'model.xml'))
    assert os.path.exists(os.path.join(export_params.save_model_to, 'model.bin'))

    ann_file = os.path.join(ote_root, 'data', 'airport', 'annotation_faces_train.json')
    data_root = os.path.join(ote_root, 'data', 'airport')

    log_file = tempfile.mktemp()
    with open(log_file, 'w') as f:
        run(f"python {MMDETECTION_DIR}/tools/test_exported.py "
            f"{env_params.config_path} "
            f"{os.path.join(export_params.save_model_to, 'model.xml')} "
            "--eval bbox "
            f"--update_config data.test.ann_file={ann_file} data.test.img_prefix={data_root}",
            shell=True, check=True, stderr=f, stdout=f)

    assert collect_ap(log_file)[0] == 0.138


def test_loading_weights(download_face_detection_0200):
    env_params = BaseTaskParameters.BaseEnvironmentParameters()
    ote_root = os.path.join(os.path.dirname(__file__), '..', '..')
    env_params.config_path = os.path.join(ote_root, 'models', 'object_detection', 'model_templates',
                                          'face-detection', 'face-detection-0200', 'model.py')
    env_params.snapshot_path = download_face_detection_0200
    env_params.work_dir = tempfile.mkdtemp()
    eval_params = BaseTaskParameters.BaseEvaluationParameters()
    task = Task(env_params, load_snapshot=False)

    task2 = Task(env_params, load_snapshot=True)

    task.load_model_from_bytes(task2.get_model_bytes())

    ann_file = os.path.join(ote_root, 'data', 'airport', 'annotation_faces_train.json')
    data_root = os.path.join(ote_root, 'data', 'airport')

    test_data_source = Dataset(ann_file, data_root)

    _, eval_results = task.test(test_data_source, eval_params)
    assert eval_results['bbox_mAP'] == 0.138
