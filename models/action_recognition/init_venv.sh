#!/usr/bin/env bash
set -v
set -x

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

CUDA_HOME_CANDIDATE=/usr/local/cuda
if [ -z "${CUDA_HOME}" ] && [ -d ${CUDA_HOME_CANDIDATE} ]; then
  echo "Exporting CUDA_HOME as ${CUDA_HOME_CANDIDATE}"
  export CUDA_HOME=${CUDA_HOME_CANDIDATE}
fi

# Download mmaction2
git submodule update --init --recursive --recommend-shallow ../../external/mmaction2

# Create virtual environment
virtualenv ${venv_dir} -p python3 --prompt="(action)"

path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino_2021}/bin/setupvars.sh"
if [[ -e "${path_openvino_vars}" ]]; then
  echo ". ${path_openvino_vars}" >> ${venv_dir}/bin/activate
fi

. ${venv_dir}/bin/activate

# upgrade pip to the latest version
pip install -U pip

cat requirements.txt | xargs -n 1 -L 1 pip3 install

mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/intel/openvino_2021}/deployment_tools/model_optimizer/requirements_onnx.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -r ${mo_requirements_file}
else
  echo "[WARNING] Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi

pip install -e ../../external/mmaction2/
MMACTION_DIR=`realpath ../../external/mmaction2/`
echo "export MMACTION_DIR=${MMACTION_DIR}" >> ${venv_dir}/bin/activate

# install ote
pip install -e ../../ote/

# check PyPI conflicts and print installed python packages
pip check
pip freeze

deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
