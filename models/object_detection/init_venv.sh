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

# Download mmdetection and nncf submodules
git submodule update --init ../../external/mmdetection
git submodule update --init ../../external/nncf

# Create virtual environment
virtualenv ${venv_dir} -p python3 --prompt="(detection)"

path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/bin/setupvars.sh"
if [[ -e "${path_openvino_vars}" ]]; then
  echo ". ${path_openvino_vars}" >> ${venv_dir}/bin/activate
fi

. ${venv_dir}/bin/activate

cat requirements.txt | xargs -n 1 -L 1 pip3 install

mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/deployment_tools/model_optimizer/requirements_onnx.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -qr ${mo_requirements_file}
else
  echo "[WARNING] Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi

pip install -e ../../external/mmdetection/
MMDETECTION_DIR=`realpath ../../external/mmdetection/`
echo "export MMDETECTION_DIR=${MMDETECTION_DIR}" >> ${venv_dir}/bin/activate

# install ote
pip install -e ../../ote/

deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
