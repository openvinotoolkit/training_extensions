import logging
import os
import re
import sys

from pylint.lint import Run

if __name__ == '__main__':
    ignored_patterns = [
        'data/',
        'external/',
        'misc/tensorflow_toolkit/',
        'misc/pytorch_toolkit/action_recognition/',
        'misc/pytorch_toolkit/face_antispoofing/',
        'misc/pytorch_toolkit/formula_recognition/',
        'misc/pytorch_toolkit/human_pose_estimation/',
        'misc/pytorch_toolkit/machine_translation/',
        'misc/pytorch_toolkit/open_closed_eye/',
        'misc/pytorch_toolkit/question_answering/',
        'misc/pytorch_toolkit/segthor/',
        'misc/pytorch_toolkit/super_resolution/',
        'misc/pytorch_toolkit/time_series/',
        'misc/pytorch_toolkit/utils/pytorch_to_onnx.py',
        'misc/tools/downscale_images.py',
        '.*venv/*',
    ]

    to_pylint = []
    wd = os.path.abspath('.')
    for root, dirnames, filenames in os.walk(wd):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, wd)
                if all(not re.match(pattern, rel_path) for pattern in ignored_patterns):
                    to_pylint.append(rel_path)

    msg_status = Run(to_pylint, exit=False).linter.msg_status
    if msg_status:
        logging.error(f'pylint failed with code {msg_status}')
        sys.exit(msg_status)
