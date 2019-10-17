#!/usr/bin/env bash

set -e

repo_dir=$(git rev-parse --show-toplevel)

# Checking installed requred tools
if ! which 2to3 >/dev/null; then
    sudo apt-get install 2to3
fi

if ! which protoc >/dev/null; then
    sudo apt-get install protobuf-compiler
fi

# Download submodules
git submodule update --init --recommend-shallow ${repo_dir}/external/cocoapi ${repo_dir}/external/models

# Build PyCocoAPI
2to3 ${repo_dir}/external/cocoapi -w
pushd ${repo_dir}/external/cocoapi/PythonAPI
make install
popd

# Prepare protobuf files
pushd ${repo_dir}/external/models/research/
protoc object_detection/protos/*.proto --python_out=.
popd
