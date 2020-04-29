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

import os
import re
import sys
import codecs
import setuptools
import glob
import sysconfig

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


INSTALL_REQUIRES = ["ninja",
                    "addict",
                    "pillow==6.2.1",
                    "texttable",
                    "scipy==1.3.2",
                    "pyyaml",
                    "networkx",
                    "graphviz",
                    "jsonschema",
                    "pydot",
                    "tensorboardX",
                    "jstyleson",
                    "matplotlib==3.0.3",
                    "numpy",
                    "tqdm",
                    "onnx",
                    "opencv-python",
                    "pytest-mock",
                    "prettytable",
                    "mdutils",
                    "yattag",
                    "jsonschema",
                    "wheel"]

DEPENDENCY_LINKS = []
if "--cpu-only" in sys.argv:
    INSTALL_REQUIRES.extend(["torch", "torchvision"])
    if sys.version_info[:2] == (3, 5):
        DEPENDENCY_LINKS = [
            'https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp35-cp35m-linux_x86_64.whl',
            'https://download.pytorch.org/whl/cpu/torchvision-0.4.2%2Bcpu-cp35-cp35m-linux_x86_64.whl']
    elif sys.version_info[:2] == (3, 6):
        DEPENDENCY_LINKS = [
            'https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp36-cp36m-linux_x86_64.whl',
            'https://download.pytorch.org/whl/cpu/torchvision-0.4.2%2Bcpu-cp36-cp36m-linux_x86_64.whl']
    elif sys.version_info[:2] >= (3, 7):
        DEPENDENCY_LINKS = [
            'https://download.pytorch.org/whl/cpu/torch-1.3.1%2Bcpu-cp37-cp37m-linux_x86_64.whl',
            'https://download.pytorch.org/whl/cpu/torchvision-0.4.2%2Bcpu-cp37-cp37m-linux_x86_64.whl']
    else:
        print("Only Python > 3.5 is supported")
        sys.exit(0)
    KEY = ["CPU"]
    sys.argv.remove("--cpu-only")
else:
    INSTALL_REQUIRES.extend(["torch==1.3.1", "torchvision==0.4.2"])
    KEY = ["GPU"]


EXTRAS_REQUIRE = {
    "tests": [
        "pytest"],
    "docs": []
}

package_data = {'nncf': ['quantization/cpu/functions_cpu.cpp',
                         'quantization/cuda/functions_cuda.cpp',
                         'quantization/cuda/functions_cuda_kernel.cu',
                         'binarization/cpu/functions_cpu.cpp',
                         'binarization/cuda/functions_cuda.cpp',
                         'binarization/cuda/functions_cuda_kernel.cu']}


setuptools.setup(
    name="nncf",
    version=find_version(os.path.join(here, "nncf/version.py")),
    author="Intel",
    author_email="alexander.kozlov@intel.com",
    description="Neural Networks Compression Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opencv/openvino-training-extensions",
    packages=setuptools.find_packages(),
    dependency_links=DEPENDENCY_LINKS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data=package_data,
    keywords=KEY
)

path_to_ninja = glob.glob(str(sysconfig.get_paths()["purelib"]+"/ninja*/ninja/data/bin/"))
if path_to_ninja:
    path_to_ninja = str(path_to_ninja[0]+"ninja")
    if not os.access(path_to_ninja, os.X_OK):
        os.chmod(path_to_ninja, 755)
