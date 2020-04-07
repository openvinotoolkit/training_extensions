import json
import os
import unittest

from common import replace_text_in_file, collect_ap


class PublicModelsTestCase(unittest.TestCase):
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

    def run_test(self, config_path, snapshot):
        print('\n\ntesting ' + config_path)
        name = config_path.replace('../../external/mmdetection/configs', '')[:-3]
        test_dir = f'/tmp/{name}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp  {config_path} {target_config_path}')
        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        os.system(f'python ../../external/mmdetection/tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval bbox  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{name}.json') as read_file:
            content = json.load(read_file)

        self.assertEqual(content['map'], ap[0])

    def download_if_not_yet(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = os.path.join(self.snapshots_dir, os.path.basename(url))
        if not os.path.exists(path):
            os.system(f'wget  {url} -P {self.snapshots_dir}')
        return path

    def test_ms_rcnn__ms_rcnn_r50_caffe_fpn_1x(self):
        origin_config = '../../external/mmdetection/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'
        url = f'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/' \
              f'ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_cascade_mask_rcnn_r50_fpn_1x(self):
        origin_config = '../../external/mmdetection/configs/cascade_mask_rcnn_r50_fpn_1x.py'
        url = f'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              f'cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_mask_rcnn_r50_caffe_c4_1x(self):
        origin_config = '../../external/mmdetection/configs/mask_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'mask_rcnn_r50_caffe_c4_1x-02a4ad3b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_cascade_rcnn_r50_caffe_c4_1x(self):
        origin_config = '../../external/mmdetection/configs/cascade_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'cascade_rcnn_r50_caffe_c4_1x-7c85c62b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_retinanet_r50_fpn_1x(self):
        origin_config = '../../external/mmdetection/configs/retinanet_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_ssd300_coco(self):
        origin_config = '../../external/mmdetection/configs/ssd300_coco.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_htc__htc_r50_fpn_1x(self):
        origin_config = '../../external/mmdetection/configs/htc/htc_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/' \
              'htc_r50_fpn_1x_20190408-878c1712.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_faster_rcnn_r50_caffe_c4_1x(self):
        origin_config = '../../external/mmdetection/configs/faster_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'faster_rcnn_r50_caffe_c4_1x-75ecfdfa.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

if __name__ == '__main__':
    unittest.main()
