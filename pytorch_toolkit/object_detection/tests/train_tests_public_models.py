import json
import os
import unittest

from common import replace_text_in_file, collect_ap


def download_if_not_yet(url, snapshots_dir):
    os.makedirs(snapshots_dir, exist_ok=True)
    path = os.path.join(snapshots_dir, os.path.basename(url))
    if not os.path.exists(path):
        os.system(f'wget  {url} -P {snapshots_dir}')
    return path


class Class(unittest.TestCase):
    coco_dir = '/tmp/coco'
    snapshots_dir = '/tmp/snapshots'

    @staticmethod
    def shorten_annotation(path, num_images):
        with open(path) as read_file:
            content = json.load(read_file)
            selected_indexes = [item['id'] for item in
                                sorted(content['images'], key=lambda x: x['id'])]
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017.zip')):
            os.system(f'wget http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017')):
            os.system(f'unzip {os.path.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, "annotations_trainval2017.zip")):
            os.system(
                f'wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            os.system(
                f'unzip {os.path.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}')

        cls.shorten_annotation(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               100)

    def test_ms_rcnn_r50_caffe_fpn_1x(self):
        model_name = 'ms_rcnn_r50_caffe_fpn_1x'
        origin_config = f'../../external/mmdetection/configs/ms_rcnn/{model_name}.py'
        test_dir = f'/tmp/{os.path.basename(origin_config)[:-3]}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp  {origin_config} {target_config_path}')

        url = f'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/' \
              f'ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'
        snapshot = download_if_not_yet(url, self.snapshots_dir)

        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        os.system(f'python ../../external/mmdetection/tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval bbox  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{model_name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])

    def test_cascade_mask_rcnn_r50_fpn_1x(self):
        model_name = 'cascade_mask_rcnn_r50_fpn_1x'
        origin_config = f'../../external/mmdetection/configs/{model_name}.py'
        test_dir = f'/tmp/{os.path.basename(origin_config)[:-3]}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp  {origin_config} {target_config_path}')

        url = f'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              f'cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth'
        snapshot = download_if_not_yet(url, self.snapshots_dir)

        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        os.system(f'python ../../external/mmdetection/tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval bbox  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{model_name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])

    def test_mask_rcnn_r50_caffe_c4_1x(self):
        model_name = 'mask_rcnn_r50_caffe_c4_1x'
        origin_config = f'../../external/mmdetection/configs/{model_name}.py'
        test_dir = f'/tmp/{os.path.basename(origin_config)[:-3]}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp  {origin_config} {target_config_path}')

        url = f'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              f'mask_rcnn_r50_caffe_c4_1x-02a4ad3b.pth'
        snapshot = download_if_not_yet(url, self.snapshots_dir)

        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        os.system(f'python ../../external/mmdetection/tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval bbox  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{model_name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])

    def test_cascade_rcnn_r50_caffe_c4_1x(self):
        model_name = 'cascade_rcnn_r50_caffe_c4_1x'
        origin_config = f'../../external/mmdetection/configs/{model_name}.py'
        test_dir = f'/tmp/{os.path.basename(origin_config)[:-3]}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp  {origin_config} {target_config_path}')

        url = f'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              f'cascade_rcnn_r50_caffe_c4_1x-7c85c62b.pth'
        snapshot = download_if_not_yet(url, self.snapshots_dir)

        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        os.system(f'python ../../external/mmdetection/tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval bbox  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{model_name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])


if __name__ == '__main__':
    unittest.main()
