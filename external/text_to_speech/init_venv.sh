#!/usr/bin/env bash
set -v
set -x

work_dir=$(realpath "$(dirname $0)")

venv_dir=$1
PYTHON_NAME=$2

if [ -z "$venv_dir" ]; then
  venv_dir=$(realpath -m ${work_dir}/venv)
else
  venv_dir=$(realpath -m "$venv_dir")
fi

if [[ -z $PYTHON_NAME ]]; then
  # the default option -- note that the minimal version of
  # python that is suitable for this repo is python3.7,
  # whereas the default python3 may point to python3.6
  PYTHON_NAME=python3
fi

PYTHON_VERSION=$($PYTHON_NAME --version | sed -e "s/^Python \([0-9]\.[0-9]\)\..*/\1/") || exit 1
if [[ $PYTHON_VERSION != "3.7" && $PYTHON_VERSION != "3.8" && $PYTHON_VERSION != "3.9" ]]; then
  echo "Wrong version of python: '$PYTHON_VERSION'"
  exit 1
fi

cd ${work_dir}

if [[ -e ${venv_dir} ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . ${venv_dir}/bin/activate"
  exit
fi

# Create virtual environment
$PYTHON_NAME -m venv ${venv_dir} --prompt="tts"

if ! [ -e "${venv_dir}/bin/activate" ]; then
  echo "The virtual environment was not created."
  exit
fi

. ${venv_dir}/bin/activate

# Get CUDA version.
CUDA_HOME_CANDIDATE=/usr/local/cuda
if [ -z "${CUDA_HOME}" ] && [ -d ${CUDA_HOME_CANDIDATE} ]; then
  echo "Exporting CUDA_HOME as ${CUDA_HOME_CANDIDATE}"
  export CUDA_HOME=${CUDA_HOME_CANDIDATE}
fi

if [ -e "$CUDA_HOME" ]; then
  if [ -e "$CUDA_HOME/version.txt" ]; then
    # Get CUDA version from version.txt file.
    CUDA_VERSION=$(cat $CUDA_HOME/version.txt | sed -e "s/^.*CUDA Version *//" -e "s/ .*//")
  else
    # Get CUDA version from directory name.
    CUDA_HOME_DIR=`readlink -f $CUDA_HOME`
    CUDA_HOME_DIR=`basename $CUDA_HOME_DIR`
    CUDA_VERSION=`echo $CUDA_HOME_DIR | cut -d "-" -f 2`
  fi
fi

# install PyTorch and MMCV.
export TORCH_VERSION=1.8.2
export CUDA_VERSION=11.1

if [[ -z ${CUDA_VERSION} ]]; then
  echo "CUDA was not found, installing dependencies in CPU-only mode. If you want to use CUDA, set CUDA_HOME and CUDA_VERSION beforehand."
else
  # Remove dots from CUDA version string, if any.
  CUDA_VERSION_CODE=$(echo ${CUDA_VERSION} | sed -e "s/\.//" -e "s/\(...\).*/\1/")
  echo "Using CUDA_VERSION ${CUDA_VERSION}"
  if [[ "${CUDA_VERSION_CODE}" != "111" ]] && [[ "${CUDA_VERSION_CODE}" != "102" ]] ; then
    echo "CUDA version must be either 11.1 or 10.2"
    exit 1
  fi
  echo "export CUDA_HOME=${CUDA_HOME}" >> ${venv_dir}/bin/activate
fi

CONSTRAINTS_FILE=$(tempfile)
export PIP_CONSTRAINT=${CONSTRAINTS_FILE}

pip install --upgrade pip || exit 1

if [[ -z $CUDA_VERSION_CODE ]]; then
  export TORCH_VERSION=${TORCH_VERSION}+cpu
else
  export TORCH_VERSION=${TORCH_VERSION}+cu$CUDA_VERSION_CODE
fi

pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html || exit 1
echo torch==${TORCH_VERSION} >> ${CONSTRAINTS_FILE}

# Install other requirements.
cat requirements.txt | xargs -n 1 -L 1 pip install -c ${CONSTRAINTS_FILE} || exit 1
cat openvino-requirements.txt | xargs -n 1 -L 1 pip install -c ${CONSTRAINTS_FILE} || exit 1

cd monotonic_align && python setup.py build_ext --inplace && cd .. || exit 1

pip install -e . || exit 1

if [[ ! -z $OTE_SDK_PATH ]]; then
  pip install -e $OTE_SDK_PATH || exit 1
elif [[ ! -z $SC_SDK_REPO ]]; then
  pip install -e $SC_SDK_REPO/src/ote_sdk || exit 1
else
  echo "OTE_SDK_PATH or SC_SDK_REPO should be specified"
  exit 1
fi


deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
