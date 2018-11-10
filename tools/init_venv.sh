#!/usr/bin/env bash

cur_dir=$(realpath "$(dirname $0)/..")

cd $cur_dir
if [ -e venv ]; then
  echo "Please remove a previously virtual environment folder."
  exit
fi

virtualenv venv -p python3
. venv/bin/activate
pip install -r requirements.txt
cd external/cocoapi
2to3 . -w
cd PythonAPI
make install
