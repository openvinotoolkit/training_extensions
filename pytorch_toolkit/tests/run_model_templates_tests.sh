# This script should be run from the `pytorch_toolkit` folder
# with one parameter -- work directory that will be used
# for instantiating model templates and running tests

export WORKDIR=$1
python3 tests/run_model_templates_tests.py --verbose

#python3 tests/run_model_templates_tests2.py --verbose --workdir $1
