#!/usr/bin/env bash

set -e

activate_command() {
  echo "To activate virtual environment run the following command:"
  echo "$ . venv/bin/activate"
}

work_dir=$(realpath "$(dirname $0)")
tf_dir=$(dirname $work_dir)
external_dir=$(realpath $tf_dir/../external)

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Virtual environment already exists"
  activate_command
  exit
fi

# Create virtual environment
virtualenv venv -p python3 --prompt="(tf)"

echo "export PYTHONPATH=\$PYTHONPATH:${tf_dir}" >> venv/bin/activate
echo "export PYTHONPATH=\$PYTHONPATH:${external_dir}/models/research:${external_dir}/models/research/slim" >> venv/bin/activate

. venv/bin/activate
pip install -r requirements.txt

# Install OpenVino Model Optimizer (optional)
mo_requirements_file="${INTEL_OPENVINO_DIR}/deployment_tools/model_optimizer/requirements_tf.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -r ${mo_requirements_file}
else
  echo "Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi

activate_command
