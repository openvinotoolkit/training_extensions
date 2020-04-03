#!/usr/bin/env bash -e

work_dir=$(realpath "$(dirname $0)")

cd ${work_dir}
if [[ -e venv ]]; then
  echo "Please remove a previously virtual environment folder '${work_dir}/venv'."
  exit
fi

# Create virtual environment
virtualenv venv -p python3.6 --prompt="(text_spotting) "
. venv/bin/activate
cat requirements.txt | xargs -n 1 -L 1 pip3 install

pushd venv

# Install custom pytorch
# https://github.com/pytorch/pytorch/pull/31595
git clone --depth=1 \
    --branch enable_export_of_custom_onnx_operations_with_tuples_as_output \
    https://github.com/Ilya-Krylov/pytorch.git

pushd pytorch
git submodule update --init --recursive
python setup.py install
popd


# Install torchvision
git clone https://github.com/pytorch/vision.git
pushd vision
git checkout be6dd4720652d630e95d968be2a4e1ae62f8807e
python setup.py install
popd

# Install instance_segmentation
pushd $(git rev-parse --show-toplevel)/pytorch_toolkit/instance_segmentation
python setup.py develop build_ext
popd
popd

# install text_spotting
python setup.py develop


# Install OpenVino Model Optimizer (optional)
mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/deployment_tools/model_optimizer/requirements_tf.txt"
if [[ -e "${mo_requirements_file}" ]]; then
  pip install -qr ${mo_requirements_file}
  echo ". ${INTEL_OPENVINO_DIR}/bin/setupvars.sh" >> venv/bin/activate
else
  echo "Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
fi


echo
echo "===================================================="
echo "To start to work, you need to activate a virtualenv:"
echo "$ . venv/bin/activate"
echo "===================================================="
