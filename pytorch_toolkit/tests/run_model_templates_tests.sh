# This script should be run from the `pytorch_toolkit` folder
# with one parameter -- work directory that will be used
# for instantiating model templates and running tests

set -v
set -x

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
PYTORCH_TOOLKIT_DIR=$(dirname $SCRIPT_DIR)
WORKDIR=$(readlink -m $1)
mkdir -p $WORKDIR || exit 1

cd $PYTORCH_TOOLKIT_DIR

pip3 install -e ote/ || exit 1

path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/bin/setupvars.sh"
source "${path_openvino_vars}" || exit 1

#export WORKDIR=$WORKDIR
#python3 tests/run_model_templates_tests.py --verbose

python3 tests/run_model_templates_tests2.py --verbose --workdir $WORKDIR
