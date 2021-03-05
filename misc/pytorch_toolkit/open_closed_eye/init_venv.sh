#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}

if [[ -e venv ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . venv/bin/activate"
fi

virtualenv venv -p python3

path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/bin/setupvars.sh"
if [[ -e "${path_openvino_vars}" ]]; then
  echo ". ${path_openvino_vars}" >> venv/bin/activate
fi


. venv/bin/activate


cat requirements.txt | xargs -n 1 -L 1 pip3 install

mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/deployment_tools/model_optimizer/requirements_onnx.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -qr ${mo_requirements_file}
else
  echo "[WARNING] Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi


echo
echo "Activate a virtual environment to start working:"
echo "$ . venv/bin/activate"