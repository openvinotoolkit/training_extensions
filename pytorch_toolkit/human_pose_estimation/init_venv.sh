#!/usr/bin/env bash
set -e

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Please remove a previously virtual environment folder '${work_dir}/venv'."
  exit
fi

# Create virtual environment
virtualenv venv -p python3 --prompt="(hpe) "
echo "export PYTHONPATH=\$PYTHONPATH:${work_dir}" >> venv/bin/activate
. venv/bin/activate

cat requirements.txt | xargs -n 1 -L 1 pip3 install

# Install OpenVino Model Optimizer (optional)
mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/indel/openvino}/deployment_tools/model_optimizer/requirements_onnx.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -qr ${mo_requirements_file}
else
  echo "Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi


echo
echo "===================================================="
echo "To start to work, you need to activate a virtualenv:"
echo "$ . venv/bin/activate"
echo "===================================================="
