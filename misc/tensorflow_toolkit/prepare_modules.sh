#!/usr/bin/env bash

set -e

repo_dir=$(git rev-parse --show-toplevel)

if [[ -z $(which 2to31) ]] || [[ -z $(which protoc) ]] ; then
    echo ""
    echo "Please install required packages by follow command and run script again:"
    echo "$ sudo apt install 2to3 protobuf-compiler"
    echo ""
fi

# Download submodules
git submodule update --init --depth 1 ${repo_dir}/external/cocoapi ${repo_dir}/external/models

# Build PyCocoAPI
2to3 ${repo_dir}/external/cocoapi -w
pushd ${repo_dir}/external/cocoapi/PythonAPI
make install
popd

# Prepare protobuf files
pushd ${repo_dir}/external/models/research/
protoc object_detection/protos/*.proto --python_out=.
popd
