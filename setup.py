"""Setup file for OTX."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

import platform

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.extension import Extension
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension

    cmd_class = {"build_ext_torch_cpp_ext": BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print("Skip building ext ops due to the absence of torch.")

cmd_class["build_ext_cythonized"] = build_ext


def get_extensions():
    if platform.system() == "Windows":
        return []

    def _cython_modules():
        cython_files = [
            "src/otx/v2/adapters/torch/mmengine/modules/pipelines/transforms/cython_augments/pil_augment.pyx",
            "src/otx/v2/adapters/torch/mmengine/modules/pipelines/transforms/cython_augments/cv_augment.pyx"
        ]

        ext_modules = [
            Extension(
                cython_file.rstrip(".pyx").lstrip("src/").replace("/", "."),
                [cython_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=["-O3"],
            )
            for cython_file in cython_files
        ]

        return cythonize(ext_modules)

    extensions = []
    extensions.extend(_cython_modules())
    return extensions

setup(
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
)
