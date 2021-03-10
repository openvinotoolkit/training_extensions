# This script should be run from the `pytorch_toolkit` folder
# with one parameter -- work directory that will be used
# for instantiating model templates and running tests
WORKDIR=$(readlink -m $1)

path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino}/bin/setupvars.sh"
source "${path_openvino_vars}" || exit 1

# This is required to log every command in this bash script
# with its command line parameters
# (this is done after source-ing openvino setupvars.sh
#  to avoid excessive logging here)
set -v
set -x

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
PYTORCH_TOOLKIT_DIR=$(dirname $SCRIPT_DIR)
mkdir -p $WORKDIR || exit 1

cd $PYTORCH_TOOLKIT_DIR

pip3 install -e ote/ || exit 1
pytest ote/tests/* || exit 1
python3 tests/run_model_templates_tests.py --verbose --workdir $WORKDIR
