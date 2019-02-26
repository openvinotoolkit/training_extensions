#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Please remove a previously virtual environment folder '${work_dir}/venv'."
  exit
fi

# Create virtual environment
virtualenv venv -p python3 --prompt="(pytorch-toolbox) "
echo "export PYTHONPATH=\$PYTHONPATH:${work_dir}" >> venv/bin/activate
. venv/bin/activate
pip install -r ${work_dir}/requirements.txt

# Install OpenVino Model Optimizer (optional)
mo_requirements_file="${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/requirements_tf.txt"
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