import logging
import os
from subprocess import run, CalledProcessError

if __name__ == '__main__':
    ignored_patters = [
        'data/',
        'external/',
        'misc/tensorflow_toolkit/',
        'misc/pytorch_toolkit/action_recognition/',
        'misc/pytorch_toolkit/face_antispoofing/',
        'misc/pytorch_toolkit/formula_recognition/',
        'misc/pytorch_toolkit/human_pose_estimation/',
        'misc/pytorch_toolkit/machine_translation/',
        'misc/pytorch_toolkit/open_closed_eye/',
        'misc/pytorch_toolkit/segthor/',
        'misc/pytorch_toolkit/super_resolution',
        'misc/pytorch_toolkit/question_answering',
        'misc/pytorch_toolkit/utils/pytorch_to_onnx.py',
        'misc/tools/downscale_images.py',
        'venv/',
    ]

    to_pylint = []
    wd = os.path.abspath('.')
    for root, dirnames, filenames in os.walk(wd):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, wd)
                if all(not rel_path.startswith(pattern) for pattern in ignored_patters):
                    to_pylint.append(rel_path)

    try:
        run(['pylint'] + to_pylint, check=True)
    except CalledProcessError:
        logging.error('pylint check failed.')
