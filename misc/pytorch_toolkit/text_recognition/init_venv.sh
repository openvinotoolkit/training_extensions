#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

venv_dir=$1
if [ -z "$venv_dir" ]; then
  venv_dir=venv
fi

cd ${work_dir}

if [[ -e ${venv_dir} ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . ${venv_dir}/bin/activate"
  exit
fi

# Create virtual environment
virtualenv ${venv_dir} -p python --prompt="(text_recognition)"

. ${venv_dir}/bin/activate

pip install --upgrade pip

cat requirements.txt | xargs -n 1 -L 1 pip install

pip install -e .

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
