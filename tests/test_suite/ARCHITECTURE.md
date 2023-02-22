# OpenVINO™ Training Extensions API test suite architecture

## I. General description

The folder `otx_sdk/otx_sdk/test_suite/` contains `otx_sdk.test_suite` library that
simplifies creation of training tests for OpenVINO™ Training Extensions algo backend.

The training tests are tests that may run in some unified manner such stages as

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

Note that the second item above is absent in such pytest library as `pytest-dependency` that just
skip a test if any of the dependencies did fail or has been skipped.

To avoid repeating of the common steps between stages the results of stages should be kept in a
special cache to be re-used by the next stages.

We suppose that each test executes one test stage (also called test action).

## II. General architecture overview

Here and below we will write paths to test suite library files relatively with the folder
`otx_sdk/otx_sdk` of OpenVINO™ Training Extensions git repository, so path to this file is referred as
`test_suite/ARCHITECTURE.md`.

When we run some test that uses `test_suite` library (typically `test_otx_training.py` in some of
the algo backends) the callstack of the test looks as follows:

- Pytest framework

- Instance of a test class.
  Typically this class is defined in `test_otx_training.py` in the algo backend.
  This class contains some fixtures implementation and uses test helper (see the next item).
  The name of the class is started from `Test`, so pytest uses it as a usual test class.
  The instance is responsible on the connection between test suite and pytest parameters and
  fixtures.

- Instance of training test helper class `OTXTestHelper` from `test_suite/training_tests_helper.py`.
  The instance of the class should be a static field of the test class stated above.
  The instance controls all execution of tests.
  Also the instance keeps in its cache an instance of a test case class between runs of different
  tests (see the next item).

- Instance of a test case class.
  This instance connects all the test stages between each other and keeps in its fields results of
  all test stages between tests.
  (Since the instance of this class is kept in the cache of training test helper's instance between
  runs of tests, results of one test may be re-used by other tests.)
  Note that each test executes only one test stage.
  And note that the class of the test case is generated "on the fly" by the function
  `generate_otx_integration_test_case_class` from the file `test_suite/training_test_case.py`;
  the function

  - receives as the input the list of action classes that should be used in tests for the
    algo backend
  - and returns the class type that will be used by the instance of the test helper.

- Instance of the test stage class `OTXTestStage` from `test_suite/training_tests_stage.py`.
  The class wraps a test action class (see the next item) to run it only once.
  Also it makes validation of the results of the wrapped test action if this is required.

- Instance of a test action class.
  The class makes the real actions that should be done for a test using calls of OpenVINO™ Training Extensions interfaces.

The next sections will describe the corresponding classes from the bottom to the top.

## III. Test actions

### III.1 General description of test actions classes

The test action classes in test suite make the real work.

Each test action makes operations for one test stage. At the moment the file
`test_suite/training_tests_actions.py` contains the reference code of the following test actions
for mmdetection algo backend:

- class `OTXTestTrainingAction` -- training of a model
- class `OTXTestTrainingEvaluationAction` -- evaluation after the training
- class `OTXTestExportAction` -- export after the training
- class `OTXTestExportEvaluationAction` -- evaluation of exported model
- class `OTXTestPotAction` -- POT compression of exported model
- class `OTXTestPotEvaluationAction` -- evaluation of POT-compressed model
- class `OTXTestNNCFAction` -- NNCF-compression of the trained model
- class `OTXTestNNCFGraphAction` -- check of NNCF compression graph (work on not trained model)
- class `OTXTestNNCFEvaluationAction` -- evaluation of NNCF-compressed model
- class `OTXTestNNCFExportAction` -- export of NNCF-compressed model
- class `OTXTestNNCFExportEvaluationAction` -- evaluation after export of NNCF-compressed model

Note that these test actions are implementation for mmdetection algo backend due to historical
reasons.
But since the actions make operations using OpenVINO™ Training Extensions interface, most of test actions code may be
re-used for all algo backends.

One of obvious exceptions is the training action -- it uses real datasets for a concrete algo
backend, and since different algo backends have their own classes for datasets (and may could have a
bit different ways of loading of the datasets) the training action should be re-implemented for each
algo backends.

Note that each test action class MUST have the following properties:

- it MUST be derived from the base class `BaseOTXTestAction`;
- it MUST override the static field `_name` -- the name of the action, it will be used as a unique
  identifier of the test action and it should be unique for the algo backend;
- if validation of the results of the action is required, it MUST override the static field
  `_with_validation` and set `_with_validation = True`;
- if it depends on the results of other test actions, it MUST override the field
  `_depends_stages_names`, the field should be a list of `str` values and should contain
  all the names of actions that's results are used in this action
  (the desired order of the names could be the order how the actions should be executed, but note
  that even in the case of another order in this list the dependent actions will be executed in the
  correct order);
- (NB: the most important) it MUST override the method `__call__` -- the method should execute the
  main action of the class and return a dict that will be stored as the action results.

Please, note that the method `__call__` of an action class MUST also have the following declaration:

```python
    def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):
```

It receives as the first parameter the `DataCollector` class that allows to store some results of
execution of the action into the test system's database
(if the test is executed on our CI system, these results will be stored to the centralized database
of our CI that could be accessed through several dashboards).

Also it receives as the second parameter `results_prev_stages` -- it is an `OrderedDict` that
contains all the results of the previous stages:

- each key is a name of test action
- each value is a dict, that was returned as the result of the action.

The `__call__` method MUST return as the result a dict that will be stored as the result of the
action (an empty dict is acceptable).

**Example:**
The class `OTXTestTrainingAction` in the file `test_suite/training_tests_actions.py`
implements the training action for mmdetection, it has `_name = "training"` and its method
`__call__` returns as the result a dict

```python
        results = {
            "model_template": self.model_template,
            "task": self.task,
            "dataset": self.dataset,
            "environment": self.environment,
            "output_model": self.output_model,
        }
```

It means that the action class `OTXTestTrainingEvaluationAction` that makes evaluation after
training in its method `__call__` can use

```python
kwargs = {
    "dataset": results_prev_stages["training"]["dataset"],
    "task": results_prev_stages["training"]["task"],
    "trained_model": results_prev_stages["training"]["output_model"],
}
```

### III.2 When implementation of own test action class is required

Please, note that `test_suite/training_tests_actions.py` contains reference code of actions for
mmdetection algo backend. This is done due to historical reasons and due to fact that mmdetection is
the first algo backend used in OpenVINO™ Training Extensions.

As we stated above, fortunately, most of test actions may be re-used for other algo backends, since
to make some test action the same OpenVINO™ Training Extensions calls should be done.

But if for an algo backend some specific test action should be done, an additional test action class
could be also implemented for the algo backend (typically, in the file `test_otx_training.py` in the
folder `tests/` of the algo backend).

Also if an algo backend should make some test action in a bit different way than in mmdetection, the
test action for the algo backend should be re-implemented.

_Example:_ For MobileNet models in image classification algo backend the NNCF compression requires
loading of the secondary (auxiliary) model. (It is required since NNCF compression requires
training, and for training MobileNet models deep-object-reid algo backend uses a specific auxiliary
model as a regularizer.)

Please, note that if you re-implementing a test action class for an algo backend it is HIGHLY
RECOMMENDED that it returns as the result dict with THE SAME keys as for the original test action
class in `test_suite/training_tests_actions.py`, and, obviously, the values for the keys have the
same meaning as for the original class. It is required since other test actions could use the result
of this test action, and if you replace a test action you should keep its interface for other
actions classes -- otherwise you will have to re-implement also all the test actions classes that
depends on this one.

Also there is a case when a new test action class should be additionally implemented in
`test_suite/training_tests_actions.py` -- when we found out that addition test action should be used
for all algo backends.

### III.3 How to implement own test action class

Please, note that this section covers the topic how to implement a new test action class, but does
not cover the topic how to make the test action class to be used by tests -- it is covered below in
the section TODO[should be written].

To implement your own test action you should do as follows:

1. Create a class derived from `OTXTestTrainingAction`
2. Set in the class the field `_name` to the name of the action
3. Set in the class the field `_with_validation = True` if validation of the action results is
   required
4. Set in the class the field `_depends_stages_names` to the list of `str` values of the names of
   test actions which results will be used in this test
5. Implement a protected method of the class which makes the real work by calling OpenVINO™ Training Extensions operations
   NB: the method should receive the parameter `data_collector: DataCollector` and use it to
   store some results of the action to the CI database
   (see how the class `DataCollector` is used in several actions in
   `test_suite/training_tests_actions.py`)
6. Implement the method `__call__` of the class with the declaration
   `def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):`
   See as the reference the method `__call__` of the class `OTXTestTrainingEvaluationAction`
   from the file `test_suite/training_tests_actions.py`.
   The method should work as follows:

- call `self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)`
  (NB: this is a required step, it will allow to catch important errors if you connect several
  test actions with each other in a wrong way)
- get from the field `results_prev_stages` results of previous stages that should be used
  and convert them to the arguments of the protected method in the item 5 above
- call the protected function from the item 5 above
- the results of the method convert to a dict and return the dict from the method `__call__`
  to store them as the result of the action

## IV. Test stage class

### IV.1 General description of test stage class

The class `OTXTestStage` from `test_suite/training_tests_stage.py` works as a wrapper for a test
action. For each instance of a test action an instance of the class `OTXTestStage` is created.

It's constructor has declaration

```python
def __init__(self, action: BaseOTXTestAction, stages_storage: OTXTestStagesStorageInterface):
```

- The `action` parameter here is the instance of action that is wrapped.
  It is kept inside the `OTXTestStage` instance.
- The `stages_storage` here is an instance of a class that allows to get a stage by name, this will
  be a test case class that connects all the test stages between each other and keeps in its fields
  results of all test stages between tests
  (all the test case classes are derived from OTXTestStagesStorageInterface)

The `stages_storage` instance is also kept inside `OTXTestStage`, it will be used to get for each
stage its dependencies.
Note that the abstract interface class `OTXTestStagesStorageInterface` has the only abstract method
`get_stage` with declaration

```python
def get_stage(self, name: str) -> "OTXTestStage":
```

-- it returns test stage class by its name.

Note that test stage has the property `name` that returns the name of its action
(i.e. the name of a stage equals to the name of the wrapped action).

The class `OTXTestStage` has method `get_depends_stages` that works as follows:

1. get for the wrapped action the list of names from its field `_depends_stages_names` using the
   property `depends_stages_names`
2. for each of the name get the stage using the method `self.stages_storage.get_stage(name)`
   -- this will be a stage (instance of `OTXTestStage`) that wraps the action with the corresponding
   name.
3. Return the list of `OTXTestStage` instances received in the previous item.

As stated above, the main purposes of the class `OTXTestStage` are:

- wrap a test action class (see the next item) to run it only once, together with all its
  dependencies
- make validation of the results of the wrapped test action if this is required.

See the next sections about that.

### IV.2 Running a test action through its test stage

The class `OTXTestStage` has a method `run_once` that has the following declaration

```python
    def run_once(
        self,
        data_collector: DataCollector,
        test_results_storage: OrderedDict,
        validator: Optional[Validator],
    ):
```

The parameters are as follows:

- `data_collector` -- interface to connect to CI database, see description of the methods `__call__`
  of the actions in the section "III.1 General description of test actions classes."
- `test_results_storage` -- it is an OrderedDict where the results of the tests are kept between
  tests, see description of the parameter `results_prev_stages` in the section
  "III.1 General description of test actions classes."
- `validator` -- optional parameter, if `Validator` instance is passed, then validation may be done
  (see the next section "IV.3 Validation of action results"), otherwise validation is skipped.

The method works as follows:

1. runs the dependency chain of this stage using recursive call of `run_once` as follows:
   - Get all the dependencies using the method `OTXTestStage.get_depends_stages` described in the
     previous section -- it will be the list of other `OTXTestStage` instances.
   - For each of the received `OTXTestStage` call the method `run_once` -- it is the recursion step
     Attention: in the recursion step the method `run_once` is called with parameter
     `validator=None` to avoid validation during recursion step -- see details in the next section
     "IV.3 Validation of action results"
2. runs the action of the stage only once:
   - If it was not run earlier -- run the action
     - if the action executed successfully
       - store result of the action into `test_result_storage` parameter
       - run validation if required
       - return
     - if the action executed with exception
       - store the exception in a special field
       - re-raise the exception
   - If it was already run earlier, check if there is stored exception
     - if there is no stored exception -- it means that the actions was successful
       and its result is already stored in the `test_result_storage` parameter
       - run validation if required
         (see details in the next section)
       - return
     - if there is a stored exception -- it means that the actions was NOT successful
       - re-raise the exception

As you can see if an exception is raised during some action, all the actions that depends on this
one will re-raise the same exception.

Also as you can see if we run a test for only one action, the `run_once` call of the stage will run
actions in all the dependent stages and use their results, but when we run many tests each of the
test also will call `run_once` for all the stages in the dependency chains, but the `run_once` calls
will NOT re-run actions for the tests.

### IV.3 Validation of action results -- how it works

As stated above, one of the purposes of `OTXTestStage` is validation of results of the wrapped
action.

As you can see from the previous section the validation is done inside `run_once` method,
and the necessary (but not sufficient) condition of running validation is that `validator` parameter
of this method is not None.

The class `Validator` is also implemented in `test_suite/training_tests_stage.py` file.
It has only one public method `validate` that has the declaration

```python
    def validate(self, current_result: Dict, test_results_storage: Dict):
```

The parameters are:

- `current_result` -- the result of the current action
- `test_results_storage` -- an OrderedDict that stores results from the other actions that were run.

The method returns nothing, but may raise exceptions to fail the test.

The `Validator` compares the results of the current action with expected metrics and with results of
the previous actions. Note that results of previous actions are important, since possible validation
criteria also may be

- "the quality metric of the current action is not worse than the result of _that_ action with
  possible quality drop 1%"
- "the quality metric of the current action is the same as the result of _that_ action with
  possible quality difference 1%"

-- these criteria are highly useful for "evaluation after export" action (quality should be almost
the same as for "evaluation after training" action) and for "evaluation after NNCF compression"
action (quality should be not worse than for "evaluation after training" action with small possible
quality drop).

As we stated above in the previous section, when the method `run_once` runs the recursion to run
actions for the dependency chain of the current action, the method `run_once` in recursion step is
called with the parameter `validator=None`.

It is required since

- `Validator` does not return values but just raises exception to fail the test if the required
  validation conditions are not met
- so, if we ran dependency actions with non-empty `Validator`, then the action test would be failed
  if some validation conditions for the dependent stages are failed -- this is not what we want to
  receive, since we run the dependency actions just to receive results of these actions
- so, we do NOT do it, so we run dependency chain with `validator=None`

Also note that there is possible (but rare) case when a stage is called from dependency chain, and
only after that it is run from a test for which this action is the main action.
For this case (as we stated above in the previous section when we described how the method
`run_once` works) we may call validation (if it is required) even if the stage was already run
earlier and was successful.
Why this case is rare? because we ask users to mention dependencies in the field
`_depends_stages_names` in the order of their execution (see description of the field), so typically
the stages are run in the right order.

As we stated above the `validator is not None` is the necessary condition to run validation, but it
is not sufficient.
The list of sufficient conditions to run real validation in `run_once` is as follows:

- The parameter `validator` of `run_once` method satisfies `validator is not None`
  (i.e. the validation is run not from the dependency chain).
- For the action the field `_with_validation == True`.
  If `_with_validation == False` it means that validation for this action is impossible -- e.g.
  "export" action cannot be validated since it does not return quality metrics, but the action
  "evaluation after export" is validated.
- The current test has the parameter `usecase == "reallife"`.
  If a test is not a "reallife" test it means that a real training is not made for the test,
  so we cannot expect real quality, so validation is not done.
  See description of test parameters below in the section TODO.

To investigate in details the conditions see the declaration of constructor of the `Validator`
class:

```python
    def __init__(self, cur_test_expected_metrics_callback: Optional[Callable[[], Dict]]):
```

As you can see it receives only one parameter, and this parameter is NOT a structure that
describes the requirements for the expected metrics for the action, but the parameter is
a FACTORY that returns the structure.

It is required since

1. constructing the structure requires complicated operations and reading of YAML files,
2. if validation should be done for the current test, and the expected metrics for the tests are
   absent, the test MUST fail
   (it is important to avoid situations when developers forget to add info on expected metrics and
   due to it tests are not failed)
3. but if validation for the current test is not required the test should not try to get the
   expected metrics

So to avoid checking of expected metrics structures for the tests without validation, an algo
backend a factory is used -- the factory for an action's validator is called if and only if
the action should be validated.

The factory is implemented in the test suite as a pytest fixture -- see the fixture
`cur_test_expected_metrics_callback_fx` in the file `test_suite/fixtures.py`.

The fixture works as follows:

- receives from other fixtures contents of the YAML file that is pointed to pytest as the pytest
  parameter `--expected-metrics-file`
- checks if the current test is "reallife" training or not (if the "usecase" parameter of the test
  is set to the value "reallife"),
- if it is not reallife then validation is not required -- in this case
  - the fixture returns None,
  - the Validator class receives None as the constructor's parameter instead of a factory,
  - Validator understands it as "skip validation"
- if this is reallife training test, the fixture returns a factory function

The returned factory function extracts from all expected metrics the expected metrics for the
current test (and if the metrics are absent -- fail the current test).

### IV.4 Validation of action results -- how expected metrics are set

As stated in the previous section, a file with expected metrics for validation is passed to pytest
as an additional parameter `--expected-metrics-file`.
It should be a YAML file.
Such YAML files are stored in each algo backend in the following path
`tests/expected_metrics/metrics_test_otx_training.yml`
(the path relative w.r.t. the algo backend root)
Examples:

- `external/mmdetection/tests/expected_metrics/metrics_test_otx_training.yml`
- `external/deep-object-reid/tests/expected_metrics/metrics_test_otx_training.yml`
- `external/mmsegmentation/tests/expected_metrics/metrics_test_otx_training.yml`

The expected metric YAML file should store a dict that maps tests to the expected metric
requirements.

The keys of the dict are strings -- the parameters' part of the test id-s. This string uniquely
identifies the test, since it contains the required action, and also the description of a model, a
dataset used for training, and training parameters.

See the detailed description how the method `OTXTestHelper._generate_test_id` works in the
subsection "VI.5.5 `short_test_parameters_names_for_generating_id`" of the section
"VI.5 Methods of the test parameters interface class `OTXTestCreationParametersInterface`"

Although the id-s are unique, they have a drawback -- they are quite long, since they contain all
the info to identify the test.

Examples of such keys are:

- `ACTION-training_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife`
- `ACTION-nncf_export_evaluation,model-Custom_Image_Classification_EfficinetNet-B0,dataset-lg_chem,num_epochs-CONFIG,batch-CONFIG,usecase-reallife`

Example of the whole part of expected metrics configuration for one of mmdetection test cases

```yaml
"ACTION-training_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife":
  "metrics.accuracy.f-measure":
    "target_value": 0.81
    "max_diff_if_less_threshold": 0.005
    "max_diff_if_greater_threshold": 0.06
"ACTION-export_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife":
  "metrics.accuracy.f-measure":
    "base": "training_evaluation.metrics.accuracy.f-measure"
    "max_diff": 0.01
"ACTION-pot_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife":
  "metrics.accuracy.f-measure":
    "base": "export_evaluation.metrics.accuracy.f-measure"
    "max_diff": 0.01
"ACTION-nncf_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife":
  "metrics.accuracy.f-measure":
    "base": "training_evaluation.metrics.accuracy.f-measure"
    "max_diff_if_less_threshold": 0.01
"ACTION-nncf_export_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife":
  "metrics.accuracy.f-measure":
    "base": "nncf_evaluation.metrics.accuracy.f-measure"
    "max_diff": 0.01
```

As you can see in this example

- the target metric "metrics.accuracy.f-measure" for the action "evaluation after training" for this
  test case is `0.81` with permissible variation `[-0.005, +0.06]`
- the target metric "metrics.accuracy.f-measure" for the action "evaluation after export" should be
  the same as for the action "evaluation after training" with permissible variation `[-0.01, +0.01]`
- the target metric "metrics.accuracy.f-measure" for the action "evaluation after pot" should be
  the same as for the action "evaluation after export" with permissible variation `[-0.01, +0.01]`
- the target metric "metrics.accuracy.f-measure" for the action "evaluation after nncf" should be
  the same as for the action "evaluation after training" with permissible variation
  `[-0.01, +infinity]`

## V. Test Case class

### V.1 General description of test case class

As stated above, test case class instance connects the test stages between each other and keeps
in its fields results of the kept test stages between tests.

Since the instance of this class is kept in the cache of training test helper's instance between
runs of tests, results of one test may be re-used by other tests.

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
2. a dataset - typically it is a dataset name that should be used
   (we use known pre-defined names for the datasets on our CI)
3. other training parameters:
   - `batch_size`
   - `num_training_epochs`

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
- An instance of the test case class is created once for each of the group of tests stated above
  -- so, the instance of test case class is created for each "test case" described above.

As stated above, the instance of test case class is kept inside cache in OpenVINO™ Training Extensions Test Helper class, it
allows to use the results of the previous tests of the same test case in the current test.

### V.2 Base interface of a test case class, creation of a test case class

The class of the test case is generated "on the fly" by the function
`generate_otx_integration_test_case_class` from the file `test_suite/training_test_case.py`;
the function has the declaration

```python
def generate_otx_integration_test_case_class(
    test_actions_classes: List[Type[BaseOTXTestAction]],
) -> Type:
```

The function `generate_otx_integration_test_case_class` works as follows:

- receives as the input the list of action classes that should be used in the test case
  -- the test case will be a storage for the test stages wrapping the actions and will connect the
  test stages with each other
- and returns the class type that will be used by the instance of the test helper.

The variable with the type of test case received from the function is stored in the test helper
instance -- it is stored in a special class "test creation parameters", see about it below in the
section TODO.

Note that the result of this function is a `class`, not an `instance` of a class.
Also note that the function receives list of action `classes`, not `instances` -- the instances of
test action classes are created when the instance of the test case class is created.

The class of the test case for a test is always inherited from the abstract interface class
`OTXTestCaseInterface`.
It is derived from the abstract interface class `OTXTestStagesStorageInterface`, so it has the
abstract method `get_stage` that for a string `name` returns test stage instance with this name.

The interface class `OTXTestCaseInterface` has two own methods:

- abstract classmethod `get_list_of_test_stages` without parameters that returns the list of names
  of test stages for this test case
- abstract method `run_stage` that runs a stage with pointed name, the method has declaration

```python
    @abstractmethod
    def run_stage(self, stage_name: str, data_collector: DataCollector,
                  cur_test_expected_metrics_callback: Optional[Callable[[], Dict]]):
```

When the test case method `run_stage` is called, it receives as the parameters

- `stage_name` -- the name of the test stage to be called
- `data_collector` -- the `DataCollector` instance that is used when the method `run_once` of the
  test stage is called
- `cur_test_expected_metrics_callback` -- a factory function that returns the expected metrics for
  the current test, the factory function is used to create the `Validator` instance that will make
  validation for the current test.

The method `run_stage` of a created test case class always does the following:

1. checks that `stage_name` is a known name of a test stage for this test case
2. creates a `Validator` instance for the given `cur_test_expected_metrics_callback`
3. finds the test stage instance for the given `stage_name` and run for it `run_once` method as
   described above in the section "IV.2 Running a test action through its test stage" with the
   parameters `data_collector` and validator

If we return back to the `OTXTestCaseInterface`, we can see that the test case class derived from it
should implement the classmethod `get_list_of_test_stages` without parameters that returns the list
of names of test stages for this test case.

Note that this method `get_list_of_test_stages` is a classmethod, since it is used when pytest
collects information on tests, before the first instance of the test case class is created.

> NB: We decided to make the test case class as a class that is generated by a function instead of a
> "normal" class, since we would like to encapsulate everything related to the test case in one
> entity -- due to it the method`get_list_of_test_stages` is not a method of a separate entity, but
> a classmethod of the test case class.
> This could be changed in the future.

Also note that the function `generate_otx_integration_test_case_class` does not makes anything
really complicated for creation of a test case class: all test case classes are the same except the
parameter `test_actions_classes` with the list of action classes that is used to create test stage
wrapper for each of the test action from the list.

### V.3 The constructor of a test case class

As stated above, the function `generate_otx_integration_test_case_class` receives as a parameter
list of action `classes`, not `instances` -- the instances of test action classes are created when
the instance of the test case class is created.
That is during construction of test case class its constructor creates instances of all the actions.

Each test case class created by the function `generate_otx_integration_test_case_class` has
the following constructor:

```python
def __init__(self, params_factories_for_test_actions: Dict[str, Callable[[], Dict]]):
```

The only parameter of this constructor is `params_factories_for_test_actions` that is
a dict:

- each key of the dict is a name of a test action
- each value of the dict is a factory function without parameters that returns the
  structure with kwargs for the constructor of the corresponding action

Note that most of the test actions do not receive parameters at all -- they receive the result of
previous actions, makes its own action, may make validation, etc.

For this case if the dict `params_factories_for_test_actions` does not contain as a key the name of
an action, then the constructor of the corresponding action will be called without parameters.

The constructor works as follows:

- For each action that was passed to the function `generate_otx_integration_test_case_class` during
  creation of this test case class
  - take name of the action
  - take `cur_params_factory = params_factories_for_test_actions.get(name)`
    - if the result is None, `cur_params = {}`
    - otherwise, `cur_params = cur_params_factory()`
  - call constructor of the current action as
    `cur_action = action_cls(**cur_params)`
  - wraps the current action with the class `OTXTestStage` as follows:
    `cur_stage = OTXTestStage(action=cur_action, stages_storage=self)`
  - store the current stage instance as
    `self._stages[cur_name] = cur_stage`

As you can see for each factory in the dict `params_factories_for_test_actions` the factory is
called lazily -- it means, it is called when and only when the corresponding action should be
created.

Also as you can see the dict `params_factories_for_test_actions` with the factories is passed to the
constructor as the parameter -- so, the factories may be different for each test to pass to the test
case the values corresponding to the current test.

## VI. Test Helper class

### VI.1 General description

Training test helper class `OTXTestHelper` is implemented in `test_suite/training_tests_helper.py`.
An instance of the class controls all execution of tests and keeps in its cache an instance of a
test case class between runs of different tests.

The most important method of the class are

- `get_list_of_tests` -- allows pytest trick generating test parameters for the test class.
  When pytest collects the info on all tests, the method returns structures that allows to make
  "pytest magic" to group and reorder the tests (see details below).
- `get_test_case` -- gets an instance of the test case class for the current test parameters, allows
  re-using the instance between several tests.

Note that the both of the methods work with test parameters that are used by pytest.

### VI.2 How pytest works with test parameters

#### VI.2.1 Short description how pytest works

Since `OTXTestHelper` makes all the operations related to pytest parametrization mechanisms, we need
to describe here how pytest works with test parameters.

Generally pytest works as follows:
(NB: it is a short and may be approximate description! do not use it as a pytest documentation)

1. Pytest collects test info, for each test function or test method it gets information on
   parameters of the test and possible combination of parameters that may be executed.
2. Then pytest makes filtering -- it selects/deselects tests based on the pytest parameters
   (e.g. `-k`) and the names of the tests
   -- each test with some combination of parameters has a full name of "test with parameters" that
   uniquely identifies the test with the parameters
3. Then pytest executes the selected tests one by one.
   When pytest executes a test function or a test method it gets a concrete combinations of
   parameter values for the parameters of the test and executes the test function/method with this
   combination.
   During the execution pytest may print the full name of the "test with parameters"

#### VI.2.2 How pytest gets information on parameters

In pytest the information on test parameters for each test function/method consists of the following
3 elements:
(NB: it is a short and may be approximate description! do not use it as a pytest documentation)

1. `argnames` -- a tuple of names of parameters of the test, typically this is a short tuple of
   strings
   - its length is the number of parameters of the test,
   - it contains string names of the parameters
2. `argvalues` -- a list of parameters of the test, this is a long list,
   - its length is the number of different combination of parameter values for the test,
   - each element of the list should be a tuple,
   - the length of each of the tuples is the same as the length of `argnames` above,
   - the tuple stores a concrete combination of values of the parameters
3. `ids` -- a list of string identifiers,
   - the list has the same length as the list `argvalues`
   - each value is a string
   - the string is used as an ID of the concrete combination of parameters
     particularly, this parameters ID is used when pytest generates the full name of the
     "test with parameters"
     (as stated above it is required for printing the full name or when some filtering is made in
     pytest on full test names)
     -- note that usually this full name in pytest looks as
     `test_name + "[" + parameters_ID + "]"`

Usually pytest collects this information inside itself, but our test suite uses special interface
that allows to change it: if pytest finds the function `pytest_generate_tests` with declaration

```python
def pytest_generate_tests(metafunc):
```

then special "pytest magic" is allowed. This 'pytest magic" allows sets for a concrete test
function/method the three elements stated above.

See a bit more details how this pytest magic works in the description of the function
`otx_pytest_generate_tests_insertion` below in the section TODO.

#### VI.2.3 How pytest runs a test with a combination of parameters

When pytest runs a test function/method that has some parameters, pytest works as follows:
(NB: it is a short and may be approximate description! do not use it as a pytest documentation)

1. gets the triplet `argnames, argvalues, ids` for this test function/method
2. check that the test function/method has all the parameters with names from the tuple `argnames`
3. makes filtering (selecting/deselecting) of concrete parameter values combinations as on pairs of
   `zip(argvalues, ids)` based on `ids` string identifiers and different pytest command line
   arguments (see pytest option `-k`)
4. for each selected combination of parameter values -- a pair `(arvalue_el, id)` from
   `zip(argvalues, ids)` -- do the following:
   - check that `argvalue_el` is a tuple with the length equal to `argnames`
   - create kwargs dict for the test function/method
   - sets in the kwargs dict for each key from `argnames` the corresponding value from
     `argvalue_el` probably in the following manner:
     `for i in range(len(argnames)): kwargs[argnames[i]] = argvalue_el[i]`

### VI.3 How pytest parametrization mechanisms relates to the test suite and `OTXTestHelper`

**(IMPORTANT)** The description how pytest works with test functions/methods parametrization in the
previous section relates to all pytest-based code.
But we would like to describe some important points related to `OTXTestHelper` and the test suite as
a whole:

- typically for one OpenVINO™ Training Extensions task type for all training tests there is only one test class with only only
  one test method that has a lot of combination of test parameters values
- the method `get_list_of_tests` of `OTXTestHelper` returns this triplet
  `argnames, argvalues, ids` that is used later in `pytest_generate_tests`-related pytest magic to
  parametrize this test method
  Note that the triplet `argnames, argvalues, ids` received from `get_list_of_tests` is used as is
  without any changes.
- `OTXTestHelper` always defines `argnames = ("test_parameters",)`, so formally the only test method
  uses **only one** test parameter to parametrise tests, but values of the parameter are dict-s that
  contain info on real test parameters

### VI.4 Constructor of the class `OTXTestHelper`

The constructor of the class `OTXTestHelper` has the following declaration

```python
def __init__(self, test_creation_parameters: OTXTestCreationParametersInterface):
```

As you can see it receives as the only parameter the class that is derived from
`OTXTestCreationParametersInterface`.
We will refer to it as a _test parameters class_ and we will refer to the base class
`OTXTestCreationParametersInterface` as to _test parameters interface_.

We suppose that such test parameter class derived from `OTXTestCreationParametersInterface` contains
most of information required to connect the test suite with a concrete algo backend.
All the methods of the interface class are abstract methods without parameters that return
structures making this connection.

Example of such implementation is the class `DefaultOTXTestCreationParametersInterface` that
contains implementation of almost all the test parameter class methods for mmdetection algo backend
(mmdetection is chosen due to historical reasons).
Nevertheless, although these methods are implemented for mmdetection, most of them may
be used without modification (or with only slight modification) for other algo backends.

The constructor of the class `OTXTestHelper` indeed makes the following:

- calls the methods of the received parameter class instance and stores the info received as
  the result of the calls in the `OTXTestHelper` instance fields
- check that the info stored in `OTXTestHelper` instance fields has a proper structure
- initialize a cache to store a test case class

### VI.5 Methods of the test parameters interface class `OTXTestCreationParametersInterface`

Let's consider all the methods of the abstract test parameters interface class one by one:

- `test_case_class`
- `test_bunches`
- `default_test_parameters`
- `test_parameters_defining_test_case_behavior`
- `short_test_parameters_names_for_generating_id`

#### VI.5.1 `test_case_class`

```python
@abstractmethod
def test_case_class(self) -> Type[OTXTestCaseInterface]:
```

The method returns a class that will be used as a Test Case class for training tests.
Note that it should return a class itself (not an instance of the class).

Typically OpenVINO™ Training Extensions Test Case class should be generated by the function
`generate_otx_integration_test_case_class` and the only parameter of the function is the list of all
test action classes that should be used in the training tests for the algo backend.

See details above in the section "V. Test Case class"

#### VI.5.2 `test_bunches`

This is the most important method since it defines the scope of the tests.

```python
@abstractmethod
def test_bunches(self) -> List[Dict[str, Any]]:
```

The method returns a test bunches list, it defines the combinations of test parameters for
which the test suite training test should be run.

The method should return a list of dicts, each of the dicts defines one test case -- see description
how test cases are defined in the section "V.1 General description of test case class".
We will call such a dict _"a test bunch dict"_ or just a _"test bunch"_.

All keys of the test bunch dicts are strings.

**(IMPORTANT)**
As stated above in "VI.3 How pytest parametrization mechanisms relates to the test suite and
`OTXTestHelper`" typically an algo backend for training tests has only one test class with only one
test method.
Note that in a typical situation a test bunch dict is passed to the only test method of the training
test class as the value `test_parameters` -- see again the section
"VI.3 How pytest parametrization mechanisms relates to the test suite and `OTXTestHelper`"

Mandatory keys of the test bunch dicts are:

- `"model_name"` -- the value is a string that is the name of a model to work with as it is defined
  in the template.yaml file of the model
- `"dataset_name"` -- the value is a string that is the name of the dataset, note that we use known
  pre-defined names for the datasets on our CI
- `"usecase"` -- the value is a string, if it is equal to `REALLIFE_USECASE_CONSTANT="reallife"`
  then validation will be run for the tests

Also typical non-mandatory keys of the test bunch dicts are

- `"num_training_iters"` or `"num_training_epochs"` or `"patience"` -- integer parameter
  restricting the training time
- `"batch_size"` -- integer parameter, affects training speed and quality

Note that the following additional tricks are used:

1. For the mandatory fields `"model_name"` and `"dataset_name"` the value may be not only a string,
   but a list of strings -- in this case a Cartesian product of all possible pairs
   `(model, dataset)` is used.
   This is the reason why this method is called `test_bunches` -- since each element of the returned
   list may define a "bunch" of tests
2. If a non-mandatory key in a test bunch dict is absent or equals to a string
   `DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST`, then it may be replaced by the corresponding default
   value pointed by the method `default_test_parameters`
   (see about it below in the section "VI.5.3 `default_test_parameters`")

Note that also most of actions that make real training (e.g. `OTXTestTrainingAction`) use one more
additional trick: if values either for `batch_size` key or for `num_training_iters` key in a test
bunch dict contain a string constant `KEEP_CONFIG_FIELD_VALUE="CONFIG"` instead of an integer value,
the action reads the values of such parameters from the template file of the model or internal
config of the model and do not change them.
It is important when we want to keep some training parameters "as is" for reallife tests and do not
want to point our own values for them.

Example of a test bunch that could be in `external/mmdetection/tests/test_otx_training.py`

```
[
    dict(
        model_name=[
           'Custom_Object_Detection_Gen3_ATSS',
           'Custom_Object_Detection_Gen3_SSD',
        ],
        dataset_name='dataset1_tiled_shortened_500_A',
        usecase='precommit',
    ),
    ...
    dict(
        model_name=[
           'Custom_Object_Detection_Gen3_ATSS',
           'Custom_Object_Detection_Gen3_SSD',
        ],
        dataset_name=[
            'bbcd',
            'weed-coco',
            'pcd',
            'aerial',
            'dice',
            'fish',
            'vitens',
            'diopsis',
        ],
        num_training_iters=KEEP_CONFIG_FIELD_VALUE,
        batch_size=KEEP_CONFIG_FIELD_VALUE,
        usecase=REALLIFE_USECASE_CONSTANT,
    )
]
```

-- in this example

- the first test bunch will make test suite to run tests for two models (ATDD and SSD) on the
  dataset `dataset1_tiled_shortened_500_A` with non-reallife training with the default `batch_size`
  and `num_training_iters`
- the second test bunch will will make test suite to run tests for two models (ATDD and SSD) on 8 of
  datasets (all pairs `model,dataset` will be run) with reallife training with the `batch_size` and
  `num_training_iters` from the original template config.

#### VI.5.3 `default_test_parameters`

```python
@abstractmethod
def default_test_parameters(self) -> Dict[str, Any]:
```

The method returns a dict that points for test parameters the default values.
The dict should have the following structure:

- each key is a string, it is a possible key in a test bunch dict
- each value is the default value for this test bunch dict key (typically, for `"batch_size"`
  and `"num_training_iters"` it is integer).

During construction of a test helper class its call the method `default_test_parameters` and stores
it to an inner field -- _default value dict_.

When a test helper instance prepares the triplet `argnames, argvalues, id` for the training test
parametrization, it does it using as the base the value received from the method `test_bunches`
-- see above in the section "VI.5.2 `test_bunches`".
As stated above in those section, during this preparation sometimes it fills some fields in the test
bunch dict by the default values.

In details, test helper in this case works as follows:

- get the default values dict received from the call of the method `default_test_parameters` of the
  test parameter class
- for each key in the dict
  - if the key is absent in the test bunch dict, or the test bunch dict contains for the key value
    `"DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST"`, then
    - set in the test bunch dict for the key the value from the default value dict

After that test helper continue work with test bunch dict as if the values always were here.

#### VI.5.4 `test_parameters_defining_test_case_behavior`

```python
@abstractmethod
def test_parameters_defining_test_case_behavior(self) -> List[str]:
```

The method returns a list of strings -- names of the test parameters
(i.e. keys of test bunches dicts) that define test case behavior.

See the detailed explanation on test cases and parameters defining test case in the section
"V.1 General description of test case class".

When several test cases are handled, if the next test has these parameters
the same as for the previous test, the test case class is re-used for the next test.
This is what allows re-using the result of previous test stages in the next test stages.

#### VI.5.5 `short_test_parameters_names_for_generating_id`

```python
@abstractmethod
def short_test_parameters_names_for_generating_id(self) -> OrderedDict:
```

This method returns an `OrderedDict` that is used to generate the `ids` part of the triplet
`argnames, argvalues, ids` that is returned by the OpenVINO™ Training Extensions test helper method `get_list_of_tests` for
the training test parametrization.

The returned OrderedDict has the following structure

- each key is a string that is a key of test bunch dicts that should be used for generating id-s
- each value is a short name of this key that will be used as an alias for string id-s generating

In details, for each combination of test parameters the string identifier `id` for the parameters'
combination is generated by the method `OTXTestHelper._generate_test_id` that is equivalent to the
following one:

```python
    def _generate_test_id(self, test_parameters):
        id_parts = []
        for par_name, short_par_name in self.short_test_parameters_names_for_generating_id.items():
            id_parts.append(f"{short_par_name}-{test_parameters[par_name]}")
        return ",".join(id_parts)
```

(here `self.short_test_parameters_names_for_generating_id` is the OrderedDict stored in the
constructor)

Note that

- If a key of test bunch dicts is not present in this OrderedDict, then it will not be present in
  the string identifier.
  So it is important to have as keys all elements of the list returned by
  `test_parameters_defining_test_case_behavior` in this OrderedDict.
- Since the length of test identifiers may be an issue, it is important to have as the values of the
  OrderedDict descriptive, but short aliases.

Example of such OrderedDict for mmdetection is as follows:

```python
OrderedDict(
    [
        ("test_stage", "ACTION"),
        ("model_name", "model"),
        ("dataset_name", "dataset"),
        ("num_training_iters", "num_iters"),
        ("batch_size", "batch"),
        ("usecase", "usecase"),
    ]
)
```

### VI.6 How the method `OTXTestHelper.get_list_of_tests` works

As stated above, the method `get_list_of_tests` returns the triplet
`argnames, argvalues, ids` that is used later in `pytest_generate_tests`-related pytest magic to
parametrize this test method, and the triplet `argnames, argvalues, ids` received from
`get_list_of_tests` is used as is without any changes.

The method `get_list_of_tests` of the class `OTXTestHelper` works as follows:

- set `argnames = ("test_parameters",)` -- as we stated above test suite training tests always use
  one parameter for pytest test, but the values of the parameter will be a dict
- get `test_bunches` list stored earlier to a field from test parameters class in constructor
  See the detailed description in the section "VI.5.2 `test_bunches`"
- get the class `test_case_class` stored earlier to a field from test parameters class in
  constructor
  See the detailed description in the section "VI.5.1 `test_case_class`"
- initialize
  `argvalues = []`
  `ids = []`
- for each test bunch in the list:
  - take the mandatory fields `model_name` and `dataset_name` from the test bunch dict
  - create the list of pairs `(model_name, dataset_name)` to be handled:
    - if either the field `model_name` or `dataset_name` is a list, generate cartesian product of
      all possible pairs using `itertools.product`
    - otherwise just take one pair `(model_name, dataset_name)`
  - for each pair `(model_name, dataset_name)`
    - for each test action name received from `test_case_class.get_list_of_test_stages()`
      - make deepcopy of the test bunch dict
      - set the key `"test_stage"` in the copied dict to the current test action name
      - set the keys `model_name` and `dataset_name` in the copied dict to the current model name
        and dataset name
      - make filling of the default values in the copied test bunch dict
        -- see the detailed description how it is done above in the subsection
        "VI.5.3 `default_test_parameters`" of the section
        "VI.5 Methods of the test parameters interface class `OTXTestCreationParametersInterface`"
      - generate the string id that corresponds to this combination of parameters using the method
        `OTXTestHelper._generate_test_id`
        -- see the detailed description how this method works in the subsection
        "VI.5.5 `short_test_parameters_names_for_generating_id`" of the section
        "VI.5 Methods of the test parameters interface class `OTXTestCreationParametersInterface`"
      - append to `argvalues` the current copied-and-modified dict
      - append to `ids` the generated string id
- when exit from all the cycles, return the triplet `argnames, argvalues, ids`

What is the result of this function?

As we stated above in the section "V.1 General description of test case class" to work properly the
test suite tests should be organized as follows:

> - The tests are grouped such that the tests from one group have the same parameters from the list
>   of "parameters that define the test case" -- it means that the tests are grouped by the
>   "test cases"
> - After that the tests are reordered such that
>   - the test from one group are executed sequentially one-by-one, without tests from other group
>     between tests in one group
>   - the test from one group are executed sequentially in the order defined for the test actions
>     beforehand;

Since for an algo backend we typically have only one test class for the training tests, only one
test method in the class, and the method is parametrized by the triplet `argnames, argvalues, ids`
received from the function `get_list_of_tests`, described above, we can say that these conditions
are fulfilled.

### VI.7 How the method `OTXTestHelper.get_test_case` works

As stated above `get_test_case` -- gets an instance of the test case class for the current test
parameters, allows re-using the instance between several tests.

It has the following declaration:

```python
def get_test_case(self, test_parameters, params_factories_for_test_actions):
```

It has the following parameters:

- `test_parameters` -- the parameters of the current test, indeed it is one of elements of the list
  `argvalues` from the triplet `argnames, argvalues, ids` received from the method
  `get_list_of_tests` -- see the previous section how it is generated
- `params_factories_for_test_actions` -- this is a dict mapping action names to factories,
  generating parameters to construct the actions, it is the same as the input parameter for the test
  case class, see detailed description in the section
  "V.3 The constructor of a test case class"
  Note that this parameter is passed to the constructor of a test case class without any changes.

Also as stated above in the section "V.1 General description of test case class" to make test suite
tests work properly the following should be fulfilled:

> - An instance of the test case class is created once for each of the group of tests stated above
>   -- so, the instance of test case class is created for each "test case" described above.

Also as we stated at the bottom of the previous section, the parameters of the only test method of
the training tests are reordered in such a way that the tests from one test case are executed
sequentially, without tests from another test case between them.

And, as also was stated in the section "V.1 General description of test case class"

> We suppose that for each algo backend there is a known set of parameters that define training
> process, and we suppose that if two tests have the same these parameters, then they are belong to
> the same test case.
> We call these parameters "the parameters defining the test case".

These parameters defining test case are received by test helper instance from the method
`test_parameters_defining_test_case_behavior` of the test parameters class.

So to keep one test case class instance the method `get_test_case` of the test helper class
`OTXTestHelper` works as follows:

- get the class `test_case_class` stored earlier to a field from test parameters class in
  constructor
  See the detailed description in the section "VI.5.1 `test_case_class`"
- get the list of string `important_params = self.test_parameters_defining_test_case_behavior`
  -- get the list of names of parameters defining test case, it was stored earlier to a field from
  the test parameters class
  See the detailed description in the section "VI.5.4 `test_parameters_defining_test_case_behavior`"
- if we already created and stored in the cache some instance of the test case class,
  - check the parameters that were used during its creation:
    if for all parameters from the list `important_params` the values of
    the parameters were the same
    - if it is True -- it is the same test case, so the function just returns the stored instance of
      the test case class
- Otherwise -- that is, if either the cache does not contain created instance of test case class, or
  some parameters from the list `important_params` were changed -- tests for another test case are
  started.
  In this case the function creates a new instance of the class `test_case_class` passing to its
  constructor the parameter `params_factories_for_test_actions`

## VII. Connecting algo backend with test suite. Test class in algo backend

The direct connection between the training test in an algo backend and the test suite is made by

- Algo backend implementation of some fixtures required for test suite
  -- see about that in the next section TODO
- Insertions that is made in the special algo backend file `tests/conftest.py` that is loaded by
  pytest before starting its work -- all the pytest magic is inserted into it.
- Test parameter class that will provide parameters to connect the algo backend with the test suite
- A test case class in the file `tests/test_otx_training.py` in the algo backend

Note again that before the test class there should be implemented a test parameters class that will
provide parameters to connect the algo backend with with test suite.
It should be a class derived from the test parameters interface class
`OTXTestCreationParametersInterface`.
See details above in the sections "VI.4 Constructor of the class `OTXTestHelper`" and
"VI.5 Methods of the test parameters interface class `OTXTestCreationParametersInterface`"
As an example of the test parameters class see

- the class `ObjectDetectionTrainingTestParameters` in the file
  `external/mmdetection/tests/test_otx_training.py`
- the class `ClassificationTrainingTestParameters` in the file
  `external/deep-object-reid/tests/test_otx_training.py`
  -- the latter is more interesting, since deep-object-reid algo backend is different w.r.t. the
  mmdetection algo backend, and we implemented the default test case parameter class
  `DefaultOTXTestCreationParametersInterface` mostly for mmdetection.

Note that test class class itself contains mostly a boilerplate code that connects test suite with
pytest.
(We made out the best to decrease the number of the boilerplate code, but nevertheless it is
required.)

Also note that the test class uses a lot of fixtures implemented in test suite.

The test case class should be implemented as follows:

- The test class should be derived from the interface class `OTXTrainingTestInterface`.
  This is required to distinguish the test classes implemented for the test suite: when pytest magic
  related to the function `pytest_generate_tests` works, it checks if the current test class is a
  subclass of this interface `OTXTrainingTestInterface` and makes parametrization only in this case.
  (See details on this pytest magic above in the section
  "VI.2.2 How pytest gets information on parameters" and below in the section TODO)

  The interface class has only one abstract classmethod `get_list_of_tests` -- see on its
  implementation below.

- The test class should have a static field `helper` defined as follows:
  ```python
  helper = OTXTestHelper(<test parameters class>())
  ```
- The test class should have the following implementation of the method `get_list_of_tests`
  ```python
  @classmethod
  def get_list_of_tests(cls, usecase: Optional[str] = None):
      return cls.helper.get_list_of_tests(usecase)
  ```
- The test class should implement as its method the fixture `params_factories_for_test_actions_fx`
  that will give the parameters for actions for the current test.
  It should work as follows:

  - use other fixtures to extract info on the current test parameters and some parameters of the
    environment (e.g. the root path where datasets is placed, etc)
  - create factories generating parameters for the test actions as function closures using
    the info extracted from the fixtures
  - and the result of the fixture is the dict `params_factories_for_test_actions`
    that maps the name of each action that requires parameters to one of the factories

    **Example**: if the algo backend has two actions that require parameters in the constructors, and
    the first of the action has the name "training" and its constructor has parameters
    `def __init__(self, dataset, labels_schema, template_path, num_training_iters, batch_size):`
    then the fixture `params_factories_for_test_actions_fx` should return a dict
    `params_factories_for_test_actions` such that
    `params_factories_for_test_actions["training"]` is a function closure that returns a dict

    ```python
              return {
                  'dataset': dataset,
                  'labels_schema': labels_schema,
                  'template_path': template_path,
                  'num_training_iters': num_training_iters,
                  'batch_size': batch_size,
              }
    ```

- The test class should implement as its method the fixture `test_case_fx` that will return the test
  case from the current implementation using the test helper cache: if it is required the
  instance of the test case class is created, otherwise the cached version of the instance is used
  (See detailed description above in the section
  "VI.7 How the method `OTXTestHelper.get_test_case` works")
  This fixture should have the following implementation
  ```python
  @pytest.fixture
  def test_case_fx(self, current_test_parameters_fx, params_factories_for_test_actions_fx):
      test_case = type(self).helper.get_test_case(current_test_parameters_fx,
                                                  params_factories_for_test_actions_fx)
      return test_case
  ```
- The test class should implement as its method the fixture `data_collector_fx` that will return the
  test the `DataCollector` instance
  NB: probably this fixture should be moved to the common fixtures
  See examples in `external/mmdetection/tests/test_otx_training.py`

- The test class should implement as its method the only test method with the name `test` and the
  following implementation:
  ```python
    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             cur_test_expected_metrics_callback_fx):
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               cur_test_expected_metrics_callback_fx)
  ```

## VIII. Connecting algo backend with test suite. Pytest magic and fixtures.

## VIII.1. Connecting algo backend with test suite. Pytest magic.

As stated above in the previous section the direct connection between the training test in an algo
backend and the test suite is made, particularly, by

> - Algo backend implementation of some fixtures required for test suite
>   -- see about that in the next section TODO
> - Insertions that is made in the special algo backend file `tests/conftest.py` that is loaded by
>   pytest before starting its work -- all the pytest magic is inserted into it.

The algo backend file `tests/conftest.py` is very important, since it is loaded by pytest before
many other operations, particularly, before collecting the tests.

The file `tests/conftest.py` for algo backend should implement the following two functions

- `pytest_generate_tests` -- as we stated above in the section
  "VI.2.2 How pytest gets information on parameters" it allows to override parametrization of a test
  function/method
  This function is called for each pytest function/method and gives the possibility to parametrize the test
  through its parameter `metafunc`
- `pytest_addoption` -- the function allows to add more command line arguments to pytest,
  the values passed to the command line arguments may be read later using the pytest fixture
  `request`.
  The function is called once before parsing of pytest command line parameters.

In test suite the file `otx_sdk/otx_sdk/test_suite/pytest_insertions.py` contains implementations of
the special functions `otx_pytest_generate_tests_insertion` and `otx_pytest_addoption_insertion`
that makes all what is required for the test suite.

As the result the minimal implementation of the functions `pytest_generate_tests` and
`pytest_addoption` contain the following boilerplate code only

```python
# pytest magic
def pytest_generate_tests(metafunc):
    otx_pytest_generate_tests_insertion(metafunc)

def pytest_addoption(parser):
    otx_pytest_addoption_insertion(parser)
```

(Why we say that it is "a minimal implementation"? because the algo backend could make its own
operations in these two functions pytest, the test suite implementation of the insertions allow to
use them together with other code.)

As we can see from the implementation `otx_pytest_generate_tests_insertion`, its main operations are
as follows:
(note that this function is called for each test function/method)

- the function get the current test class using `metafunc.cls`
- if the class is None (for test functions) or is not a subclass of `OTXTrainingTestInterface`, then
  return
- otherwise make
  ```python
    argnames, argvalues, ids = metafunc.cls.get_list_of_tests(usecase)
  ```
- parametrize the current test method by the call
  ```python
    metafunc.parametrize(argnames, argvalues, ids=ids, scope="class")
  ```
  Note that the scope "class" is used, it is required.

## VIII.2. Connecting algo backend with test suite. Pytest fixtures and others.

To connect an algo backend with the test suite the following fixtures should be implemented
in the file `tests/conftest.py` of the algo backend.

- the fixture `otx_test_domain_fx` -- it should return the string name of the
  current algo backend domain
- the fixture `otx_test_scenario_fx` -- it should return the string on the
  current test scenario, usually we use the following implementation
  ```python
        @pytest.fixture
        def otx_test_scenario_fx(current_test_parameters_fx):
            assert isinstance(current_test_parameters_fx, dict)
            if current_test_parameters_fx.get('usecase') == REALLIFE_USECASE_CONSTANT:
                return 'performance'
            else:
                return 'integration'
  ```
- the fixture `otx_templates_root_dir_fx` -- it should return the absolute
  path of the folder where OpenVINO™ Training Extensions model templates are stored for this algo backend, usually it uses
  something like `osp.dirname(osp.dirname(osp.realpath(__file__)))` to get the absolute path to the
  root of the algo backend and then using knowledge of algo backend structures point to the template
  path
- the fixture `otx_reference_root_dir_fx` -- it should return the absolute
  path of the folder where the reference values for some test operations are stored (at the moment
  such folder store the reference files for NNCF compressed graphs for the model templates).

Also the following operations should be done

```python
pytest_plugins = get_pytest_plugins_from_ote()
otx_conftest_insertion(default_repository_name='otx/training_extensions/external/mmdetection')
```

The first line points to pytest additional modules from which the fixtures should be loaded -- these
may be e2e package modules and test suite fixture module.

The second line makes some operations on variables in e2e test library that is used in our CI.
