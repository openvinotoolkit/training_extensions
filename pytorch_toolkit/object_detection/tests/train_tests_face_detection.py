import json
import os
import tempfile
import unittest


def replace_text_in_file(path, replace_what, replace_by):
    with open(path) as read_file:
        content = '\n'.join([line.rstrip() for line in read_file.readlines()])
        if content.find(replace_what) == -1:
            return False
        content = content.replace(replace_what, replace_by)
    with open(path, 'w') as write_file:
        write_file.write(content)
    return True


def collect_ap(path):
    ap = []
    beginning = 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = '
    with open(path) as read_file:
        content = [line.strip() for line in read_file.readlines()]
        for line in content:
            if line.startswith(beginning):
                ap.append(float(line.replace(beginning, '')))
    return ap


def face_detection_test_case(model_name):
    class Class(unittest.TestCase):

        def setUp(self):
            self.model_name = model_name

            self.data_folder = '../../data'
            self.work_dir = tempfile.mkdtemp()
            self.configuration_file = f'./configs/{self.model_name}.py'
            os.system(f'cp {self.configuration_file} {self.work_dir}/')
            self.configuration_file = os.path.join(self.work_dir,
                                                   os.path.basename(self.configuration_file))
            self.ote_url = 'https://download.01.org/opencv/openvino_training_extensions'
            self.url = f'{self.ote_url}/models/object_detection/{self.model_name}.pth'
            os.system(f'wget {self.url} -P {self.work_dir}')

            assert replace_text_in_file(self.configuration_file, 'imgs_per_gpu=',
                                        'imgs_per_gpu=2 ,#')
            assert replace_text_in_file(self.configuration_file, 'total_epochs = 70',
                                        'total_epochs = 75')
            assert replace_text_in_file(self.configuration_file, 'data/WIDERFace',
                                        '../../data/airport')
            assert replace_text_in_file(self.configuration_file, 'work_dir =',
                                        f'work_dir = "{os.path.join(self.work_dir, "outputs")}" #')
            assert replace_text_in_file(self.configuration_file, 'train.json',
                                        'annotation_faces_train.json')
            assert replace_text_in_file(self.configuration_file, 'val.json',
                                        'annotation_faces_train.json')
            assert replace_text_in_file(self.configuration_file, 'resume_from = None',
                                        f'resume_from = "{os.path.join(self.work_dir, self.model_name)}.pth"')

            os.system(f'cat {self.configuration_file}')

        def test_fine_tuning(self):
            log_file = os.path.join(self.work_dir, 'test_fine_tuning.log')
            os.system(
                f'../../external/mmdetection/tools/dist_train.sh {self.configuration_file} 1 --validate 2>&1 |'
                f' tee {log_file}')
            ap = collect_ap(log_file)
            self.assertEqual(len((ap)), 5)
            self.assertLess(ap[0], ap[-1])

        def test_quality_metrics(self):
            log_file = os.path.join(self.work_dir, 'test_quality_metrics.log')
            os.system(
                f'python ../../external/mmdetection/tools/test.py '
                f'{self.configuration_file} '
                f'{os.path.join(self.work_dir, self.model_name + ".pth")} '
                f'--out res.pkl --eval bbox 2>&1 | tee {log_file}')
            ap = collect_ap(log_file)

            with open(f'tests/expected_outputs/{self.model_name}.json') as read_file:
                content = json.load(read_file)

            self.assertEqual(content['map'], ap[0])

    return Class


class FaceDetection0100TestCase(face_detection_test_case('face-detection-0100')):
    """ Test case for face-detection-0100 model. """


class FaceDetection0102TestCase(face_detection_test_case('face-detection-0102')):
    """ Test case for face-detection-0102 model. """


class FaceDetection0104TestCase(face_detection_test_case('face-detection-0104')):
    """ Test case for face-detection-0104 model. """


class FaceDetection0105TestCase(face_detection_test_case('face-detection-0105')):
    """ Test case for face-detection-0105 model. """


class FaceDetection0106TestCase(face_detection_test_case('face-detection-0106')):
    """ Test case for face-detection-0106 model. """
