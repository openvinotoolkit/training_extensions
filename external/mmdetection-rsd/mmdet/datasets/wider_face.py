import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from .xml_style import XMLDataset


class WIDERFaceDataset(XMLDataset):
    """
    Reader for the WIDER Face dataset in PASCAL VOC format.
    Conversion scripts can be found in
    https://github.com/sovrasov/wider-face-pascal-voc-annotations
    """
    CLASSES = ('face',)

    def __init__(self, **kwargs):
        super(WIDERFaceDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = '{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = root.find('folder').text
            img_infos.append(
                dict(id=img_id, filename=osp.join(folder, filename),
                     width=width, height=height))

        return img_infos
