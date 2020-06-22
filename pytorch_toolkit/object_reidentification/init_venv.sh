#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Please remove a previously virtual environment folder '${work_dir}/venv'."
  exit
fi

virtualenv venv -p python3 --prompt="(pytorch-toolbox) "
echo "export PYTHONPATH=\$PYTHONPATH:${work_dir}" >> venv/bin/activate
. venv/bin/activate

pip install -r ${work_dir}/requirements.txt

# Download and setup 3rd party repo
git submodule update --init --recommend-shallow ../../external/deep-object-reid
cd ../../external/deep-object-reid
pip install -r requirements.txt
python setup.py develop
cd -

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
