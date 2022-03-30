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

# install PyTorch
export TORCH_VERSION=1.8.2

# When updating torch version, check command lines at https://pytorch.org/get-started/previous-versions/
TORCH_PIP_OPTIONS="-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html"
if [[ -z ${CUDA_VERSION} ]]; then
  echo "CUDA was not found, installing dependencies in CPU-only mode. If you want to use CUDA, set CUDA_HOME or CUDA_VERSION beforehand."
  TORCH_VERSION_POSTFIX=+cpu
else
  TORCH_VERSION_POSTFIX=
  if echo -n "${CUDA_VERSION}" |egrep -q "^10\.([2-9]|[1-9][0-9]+)($|\.)" ; then
    TORCH_CUDA_VERSION=10.2
    TORCH_VERSION_POSTFIX=+cu102
  fi
  if echo -n "${CUDA_VERSION}" |egrep -q "^11\.([1-9]|[1-9][0-9]+)($|\.)" ; then
    TORCH_CUDA_VERSION=11.1
    TORCH_VERSION_POSTFIX=+cu111
  fi
  if [[ -z "${TORCH_VERSION_POSTFIX}" ]] ; then
      echo "Need CUDA 10.* (at least 10.2) or 11.* (at least 11.1) for PyTorch 1.8.2, have CUDA version ${CUDA_VERSION}"
      exit 1
  fi
  echo "Using PyTorch for CUDA ${TORCH_CUDA_VERSION} with local CUDA ${CUDA_VERSION}"
fi

CONSTRAINTS_FILE=$(tempfile)
export PIP_CONSTRAINT=${CONSTRAINTS_FILE}

pip install --upgrade pip || exit 1

pip install torch==${TORCH_VERSION}${TORCH_VERSION_POSTFIX} ${TORCH_PIP_OPTIONS} || exit 1
echo torch==${TORCH_VERSION}${TORCH_VERSION_POSTFIX} >> ${CONSTRAINTS_FILE}

# Install other requirements.
cat requirements.txt | xargs -n 1 -L 1 pip install -c ${CONSTRAINTS_FILE} || exit 1
cat openvino-requirements.txt | xargs -n 1 -L 1 pip install -c ${CONSTRAINTS_FILE} || exit 1

cd monotonic_align && python setup.py build_ext --inplace && cd .. || exit 1

pip install -e . || exit 1

# Install OTE SDK
pip install -e ../../ote_sdk/ || exit 1

deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
