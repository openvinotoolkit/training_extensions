#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}

if [[ -e venv ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . venv/bin/activate"
fi

virtualenv venv -p python3


. venv/bin/activate


cat requirements.txt | xargs -n 1 -L 1 pip3 install


echo
echo "Activate a virtual environment to start working:"
echo "$ . venv/bin/activate"
