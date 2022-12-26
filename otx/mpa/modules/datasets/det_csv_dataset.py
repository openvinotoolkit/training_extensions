# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pandas as pd
import numpy as np

from mmdet.datasets import DATASETS
from mmdet.datasets import CustomDataset
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class CSVDatasetDet(CustomDataset):

    CLASSES = CocoDataset.CLASSES  # To use pre-trained models

    def __init__(self, data_file, **kwargs):
        self.data_file = data_file
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        # Load CSVs
        df_data = pd.read_csv(self.data_file)
        df_anno = pd.read_csv(ann_file)

        # Filter images w/o annotations
        image_ids = df_anno['ImageID'].unique()
        df_sub_data = df_data.loc[df_data['ImageID'].isin(image_ids)]
        print(f'{ann_file}: {len(image_ids)} -> {len(df_anno)}')

        data_infos = []
        for _, data in df_sub_data.iterrows():
            # Get per-image annotations
            df_sub_anno = df_anno.loc[df_anno['ImageID'] == data['ImageID']]
            bboxes = []
            labels = []
            person_label = 0
            for _, anno in df_sub_anno.iterrows():
                if anno['IsOccluded']:
                    continue
                bbox = [
                    anno['Xmin'],
                    anno['Ymin'],
                    anno['Xmax'],
                    anno['Ymax']
                ]
                bboxes.append(bbox)
                labels.append(person_label)  # TODO: parse labels
            if len(bboxes) == 0:
                continue
            # Add structured annotation
            data_infos.append(
                dict(
                    filename=data['ImagePath'],
                    height=data['Height'],
                    width=data['Width'],
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                )
            )

        return data_infos
