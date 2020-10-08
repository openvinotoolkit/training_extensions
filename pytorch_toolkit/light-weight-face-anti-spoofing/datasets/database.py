"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from functools import partial

from .celeba_spoof import CelebASpoofDataset
from .casia_surf import CasiaSurfDataset
from .lcc_fasd import LccFasdDataset

def do_nothing(**args):
    pass

# import your reader and replace do_nothing with it
external_reader=do_nothing

def get_datasets(config):

    celeba_root = config.datasets.Celeba_root
    lccfasd_root = config.datasets.LCCFASD_root
    casia_root = config.datasets.Casia_root

    #set of datasets
    datasets = {'celeba_spoof_train': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=False,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_val': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=True,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_test': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=True, multi_learning=config.multi_task_learning),

                'Casia_train': partial(CasiaSurfDataset, protocol=1, dir_=casia_root,
                                    mode='train'),

                'Casia_val': partial(CasiaSurfDataset, protocol=1, dir_=casia_root,
                                    mode='dev'),

                'Casia_test': partial(CasiaSurfDataset, protocol=1, dir_=casia_root, mode='test'),

                'LCC_FASD_train': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='train'),

                'LCC_FASD_val': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val'),

                'LCC_FASD_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='test'),

                'LCC_FASD_val_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val_test'),

                'LCC_FASD_combined': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='combine_all'),

                'external_train': partial(external_reader, **config.external.train_params),

                'external_val': partial(external_reader, **config.external.val_params),

                'external_test': partial(external_reader, **config.external.test_params)}
    return datasets
