# Quick HOW TO add training tests using OpenVINO™ Training Extensions test suite

## I. Introduction to OpenVINO™ Training Extensions test suite

### I.1 General description

OpenVINO™ Training Extensions test suite allows to create training tests

The training tests are tests that may run in some unified manner such stages (or, as we also
call it, "actions") as

- training of a model,
- evaluation of the trained model,
- export or optimization of the trained model,
- and evaluation of exported/optimized model.

Typically each OpenVINO™ Training Extensions algo backend contains test file `test_otx_training.py` that allows to run the
training tests.

Note that there are a lot of dependencies between different stages of training tests: most of them
require trained model, so they depends on training stage; also for example POT optimization stage
and evaluation of exported model stage require the exported model, so export stage should be run
before, etc.

The `test_suite` library allows to create training tests such that

1. the tests do not repeat the common steps that can be re-used
2. if we point for pytest that only some test stage is required, all dependency stages are run
   automatically
3. if a stage is failed all the stage that depend on this stage are also failed.

To avoid repeating of the common steps between stages the results of stages should be kept in a
special cache to be re-used by the next stages.

We suppose that each test executes one test stage (also called test action).

At the moment we have the following test actions:

- class `"training"` -- training of a model
- class `"training_evaluation"` -- evaluation after the training
- class `"export"` -- export after the training
- class `"export_evaluation"` -- evaluation of exported model
- class `"pot"` -- POT compression of exported model
- class `"pot_evaluation"` -- evaluation of POT-compressed model
- class `"nncf"` -- NNCF-compression of the trained model
- class `"nncf_graph"` -- check of NNCF compression graph (work on not trained model)
- class `"nncf_evaluation"` -- evaluation of NNCF-compressed model
- class `"nncf_export"` -- export of NNCF-compressed model
- class `"nncf_export_evaluation"` -- evaluation after export of NNCF-compressed model

### I.2. General description of test cases

One of the most important question is when a test may re-use results of another test.
We can consider this from the following point of view.
We suppose that the test suite indeed do not make several independent tests, but make a set of
actions with several "test cases".
Since the test suite works with OpenVINO™ Training Extensions, each "test case" is considered as a situation that could be
happened during some process of work with OpenVINO™ Training Extensions, and the process may include different actions.

Since OpenVINO™ Training Extensions is focused on training a neural network and making some operations on the trained model,
we defined the test case by the parameters that define training process
(at least they defines it as much as it is possible for such stochastic process).

Usually the parameters defining the training process are:

1. a model - typically it is a name of OpenVINO™ Training Extensions template to be used
   -- this is the field `model_template_id` of the model template YAML file
2. a dataset - typically it is a dataset name that should be used
   (we use known pre-defined names for the datasets on our CI)
3. other training parameters:
   - `batch_size`
   - `num_training_epochs` or `num_training_iters`

We suppose that for each algo backend there is a known set of parameters that define training
process, and we suppose that if two tests have the same these parameters, then they are belong to
the same test case.
We call these parameters "the parameters defining the test case".

But from pytest point of view there are just a lot of tests with some parameters.

The general approach that is used to allow re-using results of test stages between test is the
following:

- The tests are grouped such that the tests from one group have the same parameters from the list
  of "parameters that define the test case" -- it means that the tests are grouped by the
  "test cases"
- After that the tests are reordered such that
  - the test from one group are executed sequentially one-by-one, without tests from other group
    between tests in one group
  - the test from one group are executed sequentially in the order defined for the test actions
    beforehand;
- An instance of a special test case class is created once for each of the group of tests stated above
  -- so, the instance of test case class is created for each "test case" described above.

The instance of the special test case class (described in the last item of the list above)
is kept inside cache in test suite, it allows to use the results of the
previous tests of the same test case in the current test.

### I.3. String names of tests

Pytest allows running parametrized test methods in test classes.

The test suite is made such that for each OpenVINO™ Training Extensions task (e.g. "object detection", "image classification",
etc) there is one test class with one test method with the name `test`, the method is parametrized
using special pytest tricks in the function `pytest_generate_tests` in the file `conftest.py` in the
folder `tests/`.

(Note that "classical" way of parametrization of a class method is using pytest decorator
`@pytest.mark.parametrize`, but we do NOT use this way, since we need to regroup tests by test cases
-- see details in the previous section.)

For each parametrized test method the pytest framework generates its name as follows:
`<NameOfClass>.<name_of_method>[<parameters_string>]`

For the test suite the test names are generated in the same way (this is the inner part of pytest
that was not changed by us), but test suite generates the `parameters_string` part.

Test suite generates the parameters string using

1. the name of the test action (aka test stage)
2. the values of the test's parameters defining test behavior
   (see the previous section "II. General description of test cases")
3. the usecase -- at the moment it is either "precommit" or "reallife"

Note that in test suite the test parameters may have "short names" that are used during generation
of the test parameters strings.
Examples of test parameters short names

- for parameter `model_name` -- `"model"`
- for parameter `dataset_name` -- `"dataset"`
- for parameter `num_training_iters` -- `"num_iters"`
- for parameter `batch_size` -- `"batch"`

So, examples of test parameters strings are

- `ACTION-training_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife`
- `ACTION-nncf_export_evaluation,model-Custom_Image_Classification_EfficinetNet-B0,dataset-lg_chem,num_epochs-CONFIG,batch-CONFIG,usecase-reallife`

The test parameters strings are used in the test suite as test id-s.
Although the id-s are unique, they have a drawback -- they are quite long, since they contain all
the info to identify the test.

## II. How To-s

### II.1 How to add a new model+dataset pair to the training tests

Let's there are implemented training tests for some OpenVINO™ Training Extensions algo backend, and we want to add
new model+dataset pair to the training test.

In this case you should do as follows:

1. Open the file with the training tests for the task type.
   Typically it has name `test_otx_training.py` and it is placed in the folder
   `external/<algo_backend_folder>/tests/`.

2. Find the class derived either from the class `OTXTestCreationParametersInterface`
   or from the class `DefaultOTXTestCreationParametersInterface`.
   There should be only one such class in the file, it should have name like
   `ObjectDetectionTrainingTestParameters`.

3. Find the method `test_bunches` in the class.
   Most probably the method creates a variable `test_bunches` with a list of dicts,
   and returns the deepcopy of the variable.

4. Make change: add to the list a new element -- dict with the following keys
   - `model_name` -- either a string with the model name or a list of strings with the model names,
     the model names should be taken from the field `model_template_id` of the model template YAML
     file
   - `dataset_name` -- either a string with the dataset name or a list of strings with the dataset names,
     we use known pre-defined names for the datasets on our CI.
     The dataset names may be taken from the YAML file `dataset_definitions.yml` in the dataset server
     of the CI.
     (If you should add a new dataset -- please, upload your dataset in the proper folder to the
     server and point the relative paths to the dataset parts to the file `dataset_definitions.yml`
     in the folder)
     Note that if `model_name` and/or `dataset_name` are lists, the test will be executed for
     all possible pairs `(model, dataset)` from Cartesian product of the lists.
   - `num_training_iters` or `max_num_epochs` or `patience` -- either integer, or a constant
     `KEEP_CONFIG_FIELD_VALUE` to keep the value from the template, or just do not add (skip) the
     key to use the default small value for the precommit tests (1 or 2)
   - `batch_size` -- either integer, or a constant `KEEP_CONFIG_FIELD_VALUE` to keep the value from
     the template, or just do not add (skip) the key to use the default small value for the
     precommit tests (1 or 2)
   - `usecase` -- either `REALLIFE_USECASE_CONSTANT` for reallife training tests or "precommit" for
     precommit tests
