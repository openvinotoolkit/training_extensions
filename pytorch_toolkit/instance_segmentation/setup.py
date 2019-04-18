"""
 Copyright (c) 2019 Intel Corporation

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

import os.path as osp

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


this_dir = osp.dirname(osp.abspath(__file__))
extensions_dir = osp.join(this_dir, 'segmentoly', 'extensions')
sources = [
    osp.join(extensions_dir, 'extensions.cpp'),
    osp.join(extensions_dir, 'nms', 'nms.cpp'),
    osp.join(extensions_dir, 'nms', 'cpu', 'nms_kernel.cpp'),
    osp.join(extensions_dir, 'roi_align', 'roi_align.cpp'),
    osp.join(extensions_dir, 'roi_align', 'cpu', 'roi_align_kernel.cpp'),
]

extension = CppExtension
macroses = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    sources_gpu = [
        osp.join(extensions_dir, 'nms', 'gpu', 'nms_kernel.cu'),
        osp.join(extensions_dir, 'roi_align', 'gpu', 'roi_align_kernel.cu'),
    ]
    extension = CUDAExtension
    sources += sources_gpu
    macroses += [("WITH_CUDA", None)]

setup(name='segmentoly',
      ext_modules=[extension(name='segmentoly.extensions._EXTRA',
                             sources=sources,
                             include_dirs=[extensions_dir],
                             define_macros=macroses)
                   ],
      cmdclass={'build_ext': BuildExtension},
      packages=find_packages()
      )
