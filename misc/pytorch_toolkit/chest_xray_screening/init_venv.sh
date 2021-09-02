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

virtualenv ${venv_dir} -p python3.7 --prompt="(chest_x-ray_screening)"


path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/bin/setupvars.sh"
if [[ -e "${path_openvino_vars}" ]]; then
  echo ". ${path_openvino_vars}" >>${venv_dir}/bin/activate
fi

. ${venv_dir}/bin/activate


cat requirements.txt | xargs -n 1 -L 1 pip3 install

pip install -e .

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
