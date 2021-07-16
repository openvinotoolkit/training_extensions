#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

venv_dir=$1
if [ -z "$venv_dir" ]; then
  venv_dir=venv
fi

cd ${work_dir}

if [[ -e venv ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . venv/bin/activate"
fi

virtualenv ${venv_dir} -p python --prompt="(chest_x-ray_screening)"


. venv/bin/activate


cat requirements.txt | xargs -n 1 -L 1 pip3 install


echo
echo "Activate a virtual environment to start working:"
echo "$ . venv/bin/activate"
