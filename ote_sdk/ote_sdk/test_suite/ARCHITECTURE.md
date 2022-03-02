# OTE SDK test suite architecture

## I. General description

The folder `ote_sdk/ote_sdk/test_suite/` contains `ote_sdk.test_suite` library that
simplifies creation of training tests for OTE algo backend.

The training tests are tests that may run in some unified manner such stages as
* training of a model,
* evaluation of the trained model,
* export or optimization of the trained model,
* and evaluation of exported/optimized model.

Typically each OTE algo backend contains test file `test_ote_training.py` that allows to run the
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
`ote_sdk/ote_sdk` of OTE git repository, so path to this file is referred as
`test_suite/ARCHITECTURE.md`.

When we run some test that uses `test_suite` library (typically `test_ote_training.py` in some of
the algo backends) the callstack of the test looks as follows:

* Pytest framework

* Instance of a test class.  
  Typically this class is defined in `test_ote_training.py` in the algo backend.  
  This class contains some fixtures implementation and uses test helper (see the next item).  
  The name of the class is started from `Test`, so pytest uses it as a usual test class.
  The instance is responsible on the connection between test suite and pytest parameters and
  fixtures.

* Instance of training test helper class `OTETestHelper` from `test_suite/training_tests_helper.py`.  
  The instance of the class should be a static field of the test class stated above.  
  The instance controls all execution of tests.
  Also the instance keeps in its cache an instance of a test case class between runs of different
  tests (see the next item).

* Instance of a test case class.  
  This instance connects all the test stages between each other and keeps in its fields results of
  all test stages between tests.  
  (Since the instance of this class is kept in the cache of training test helper's instance between
  runs of tests, results of one test may be re-used by other tests.)
  Note that each test executes only one test stage.  
  And note that the class of the test case is generated "on the fly" by the function
  `generate_ote_integration_test_case_class` from the file `test_suite/training_test_case.py`;
  the function
  * receives as the input the list of action classes that should be used in tests for the
    algo backend
  * and returns the class type that will be used by the instance of the test helper.

* Instance of the test stage class `OTETestStage` from `test_suite/training_tests_stage.py`.  
  The class wraps a test action class (see the next item) to run it only once.  
  Also it makes validation of the results of the wrapped test action if this is required.

* Instance of a test action class.  
  The class makes the real actions that should be done for a test using calls of OTE SDK interfaces.

The next sections will describe the corresponding classes from the bottom to the top.


## III. Test actions

### III.1 General description of test actions classes

The test action classes in test suite make the real work.

Each test action makes operations for one test stage. At the moment the file
`test_suite/training_tests_actions.py` contains the reference code of the following test actions
for mmdetection algo backend:
* class `OTETestTrainingAction` -- training of a model
* class `OTETestTrainingEvaluationAction` -- evaluation after the training
* class `OTETestExportAction` -- export after the training
* class `OTETestExportEvaluationAction` -- evaluation of exported model
* class `OTETestPotAction` -- POT compression of exported model
* class `OTETestPotEvaluationAction` -- evaluation of POT-compressed model
* class `OTETestNNCFAction` -- NNCF-compression of the trained model
* class `OTETestNNCFGraphAction` -- check of NNCF compression graph (work on not trained model)
* class `OTETestNNCFEvaluationAction` -- evaluation of NNCF-compressed model
* class `OTETestNNCFExportAction` -- export of NNCF-compressed model
* class `OTETestNNCFExportEvaluationAction` -- evaluation after export of NNCF-compressed model

Note that these test actions are implementation for mmdetection algo backend due to historical
reasons.
But since the actions make operations using OTE SDK interface, most of test actions code may be
re-used for all algo backends.

One of obvious exceptions is the training action -- it uses real datasets for a concrete algo
backend, and since different algo backends have their own classes for datasets (and may could have a
bit different ways of loading of the datasets) the training action should be re-implemented for each
algo backends.

Note that each test action class MUST have the following properties:
* it MUST be derived from the base class `BaseOTETestAction`;
* it MUST override the static field `_name` -- the name of the action, it will be used as a unique
  identifier of the test action and it should be unique for the algo backend;
* if validation of the results of the action is required, it MUST override the static field
  `_with_validation` and set `_with_validation = True`;
* if it depends on the results of other test actions, it MUST override the field
  `_depends_stages_names`, the field should be a list of `str` values and should contain
  all the names of actions that's results are used in this action
  (the desired order of the names could be the order how the actions should be executed, but note
  that even in the case of another order in this list the dependent actions will be executed in the
  correct order);
* (NB: the most important) it MUST override the method `__call__` -- the method should execute the
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
* each key is a name of test action
* each value is a dict, that was returned as the result of the action.

The `__call__` method MUST return as the result a dict that will be stored as the result of the
action (an empty dict is acceptable).

**Example:**
The class `OTETestTrainingAction` in the file `test_suite/training_tests_actions.py`
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
It means that the action class `OTETestTrainingEvaluationAction` that makes evaluation after
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
the first algo backend used in OTE SDK.

As we stated above, fortunately, most of test actions may be re-used for other algo backends, since
to make some test action the same OTE SDK calls should be done.

But if for an algo backend some specific test action should be done, an additional test action class
could be also implemented for the algo backend (typically, in the file `test_ote_training.py` in the
folder `tests/` of the algo backend).

Also if an algo backend should make some test action in a bit different way than in mmdetection, the
test action for the algo backend should be re-implemented.

*Example:* For MobileNet models in image classification algo backend the NNCF compression requires
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
1. Create a class derived from `OTETestTrainingAction`
2. Set in the class the field `_name` to the name of the action
3. Set in the class the field `_with_validation = True` if validation of the action results is
  required
4. Set in the class the field `_depends_stages_names` to the list of `str` values of the names of
  test actions which results will be used in this test
5. Implement a protected method of the class which makes the real work by calling OTE SDK operations  
  NB: the method should receive the parameter `data_collector: DataCollector` and use it to
  store some results of the action to the CI database  
  (see how the class `DataCollector` is used in several actions in
  `test_suite/training_tests_actions.py`)
6. Implement the method `__call__` of the class with the declaration  
  `def __call__(self, data_collector: DataCollector, results_prev_stages: OrderedDict):`
  See as the reference the method `__call__` of the class `OTETestTrainingEvaluationAction`
  from the file `test_suite/training_tests_actions.py`.
  The method should work as follows:
  * call `self._check_result_prev_stages(results_prev_stages, self.depends_stages_names)`  
    (NB: this is a required step, it will allow to catch important errors if you connect several
    test actions with each other in a wrong way)
  * get from the field `results_prev_stages` results of previous stages that should be used
    and convert them to the arguments of the protected method in the item 5 above
  * call the protected function from the item 5 above
  * the results of the method convert to a dict and return the dict from the method `__call__`
    to store them as the result of the action

## IV. Test stage class

### IV.1 General description of test stage class

The class `OTETestStage` from `test_suite/training_tests_stage.py` works as a wrapper for a test
action. For each instance of a test action an instance of the class `OTETestStage` is created.

It's constructor has declaration
```python
def __init__(self, action: BaseOTETestAction, stages_storage: OTETestStagesStorageInterface):
```

* The `action` parameter here is the instance of action that is wrapped.  
  It is kept inside the `OTETestStage` instance.
* The `stages_storage` here is an instance of a class that allows to get a stage by name, this will
  be a test case class that connects all the test stages between each other and keeps in its fields
  results of all test stages between tests
  (all the test case classes are derived from OTETestStagesStorageInterface)

The `stages_storage` instance is also kept inside `OTETestStage`, it will be used to get for each
stage its dependencies.  
Note that the abstract interface class `OTETestStagesStorageInterface` has the only abstract method
`get_stage` with declaration
```python
def get_stage(self, name: str) -> "OTETestStage":
```
-- it returns test stage class by its name.

Note that test stage has the property `name` that returns the name of its action
(i.e. the name of a stage equals to the name of the wrapped action).

The class `OTETestStage` has method `get_depends_stages` that works as follows:
1. get for the wrapped action the list of names from its field `_depends_stages_names` using the
   property `depends_stages_names`
2. for each of the name get the stage using the method `self.stages_storage.get_stage(name)`  
   -- this will be a stage (instance of `OTETestStage`) that wraps the action with the corresponding
   name.
3. Return the list of `OTETestStage` instances received in the previous item.

As stated above, the main purposes of the class `OTETestStage` are:
* wrap a test action class (see the next item) to run it only once, together with all its
  dependencies
* make validation of the results of the wrapped test action if this is required.

See the next sections about that.

### IV.2 Running a test action through its test stage

The class `OTETestStage` has a method `run_once` that has the following declaration
```python
    def run_once(
        self,
        data_collector: DataCollector,
        test_results_storage: OrderedDict,
        validator: Optional[Validator],
    ):
```
The parameters are as follows:
* `data_collector` -- interface to connect to CI database, see description of the methods `__call__`
  of the actions in the section "III.1 General description of test actions classes."
* `test_results_storage` -- it is an OrderedDict where the results of the tests are kept between
  tests, see description of the parameter `results_prev_stages` in the section
  "III.1 General description of test actions classes."
* `validator` -- optional parameter, if `Validator` instance is passed, then validation may be done  
  (see the next section "IV.3 Validation of action results"), otherwise validation is skipped.



The method works as follows:
1. runs the dependency chain of this stage using recursive call of `run_once` as follows:
   * Get all the dependencies using the method `OTETestStage.get_depends_stages` described in the
     previous section -- it will be the list of other `OTETestStage` instances.
   * For each of the received `OTETestStage` call the method `run_once` -- it is the recursion step  
     Attention: in the recursion step the method `run_once` is called with parameter
     `validator=None` to avoid validation during recursion step -- see details in the next section
     "IV.3 Validation of action results"
2. runs the action of the stage only once:
   * If it was not run earlier -- run the action
     * if the action executed successfully
       * store result of the action into `test_result_storage` parameter
       * run validation if required
       * return
     * if the action executed with exception
       * store the exception in a special field
       * re-raise the exception
   * If it was already run earlier, check if there is stored exception
     * if there is no stored exception -- it means that the actions was successful
       and its result is already stored in the `test_result_storage` parameter
       * run validation if required
         (see details in the next section)
       * return
     * if there is a stored exception -- it means that the actions was NOT successful
       * re-raise the exception

As you can see if an exception is raised during some action, all the actions that depends on this
one will re-raise the same exception.

Also as you can see if we run a test for only one action, the `run_once` call of the stage will run
actions in all the dependent stages and use their results, but when we run many tests each of the
test also will call `run_once` for all the stages in the dependency chains, but the `run_once` calls
will NOT re-run actions for the tests.


### IV.3 Validation of action results -- how it works

As stated above, one of the purposes of `OTETestStage` is validation of results of the wrapped
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
* `current_result` -- the result of the current action
* `test_results_storage` -- an OrderedDict that stores results from the other actions that were run.

The method returns nothing, but may raise exceptions to fail the test.

The `Validator` compares the results of the current action with expected metrics and with results of
the previous actions. Note that results of previous actions are important, since possible validation
criteria also may be
* "the quality metric of the current action is not worse than the result of *that* action with
  possible quality drop 1%"
* "the quality metric of the current action is the same as the result of *that* action with
  possible quality difference 1%"

-- these criteria are highly useful for "evaluation after export" action (quality should be almost
the same as for "evaluation after training" action) and for "evaluation after NNCF compression"
action (quality should be not worse than for "evaluation after training" action with small possible
quality drop).

As we stated above in the previous section, when the method `run_once` runs the recursion to run
actions for the dependency chain of the current action, the method `run_once` in recursion step is
called with the parameter `validator=None`.

It is required since
* `Validator` does not return values but just raises exception to fail the test if the required
  validation conditions are not met
* so, if we ran dependency actions with non-empty `Validator`, then the action test would be failed
  if some validation conditions for the dependent stages are failed -- this is not what we want to
  receive, since we run the dependency actions just to receive results of these actions
* so, we do NOT do it, so we run dependency chain with `validator=None`

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
* The parameter `validator` of `run_once` method  satisfies `validator is not None`
  (i.e. the validation is run not from the dependency chain).
* For the action the field `_with_validation == True`.  
  If `_with_validation == False` it means that validation for this action is impossible -- e.g.
  "export" action cannot be validated since it does not return quality metrics, but the action
  "evaluation after export" is validated.
* The current test has the parameter `usecase == "reallife"`.  
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
* receives from other fixtures contents of the YAML file that is pointed to pytest as the pytest
  parameter `--expected-metrics-file`
* checks if the current test is "reallife" training or not (if the "usecase" parameter of the test
  is set to the value "reallife"),
* if it is not reallife then validation is not required -- in this case
  * the fixture returns None,
  * the Validator class receives None as the constructor's parameter instead of a factory,
  * Validator understands it as "skip validation"
* if this is reallife training test, the fixture returns a factory function

The returned factory function extracts from all expected metrics the expected metrics for the
current test (and if the metrics are absent -- fail the current test).



### IV.4 Validation of action results -- how expected metrics are set

As stated in the previous section, a file with expected metrics for validation is passed to pytest
as an additional parameter `--expected-metrics-file`.
It should be a YAML file.  
Such YAML files are stored in each algo backend in the following path
`tests/expected_metrics/metrics_test_ote_training.yml`
(the path relative w.r.t. the algo backend root)  
Examples:
* `external/mmdetection/tests/expected_metrics/metrics_test_ote_training.yml`
* `external/deep-object-reid/tests/expected_metrics/metrics_test_ote_training.yml`
* `external/mmsegmentation/tests/expected_metrics/metrics_test_ote_training.yml`

The expected metric YAML file should store a dict that maps tests to the expected metric
requirements.

The keys of the dict are strings -- the parameters' part of the test id-s. This string uniquely
identifies the test, since it contains the required action, and also the description of a model, a
dataset used for training, and training parameters.  
(See the description of the function `OTETestHelper._generate_test_id` below in the section TODO.)  
Although the id-s are unique, they have a drawback -- they are quite long, since they contain all
the info to identify the test.

Examples of such keys are:
* `ACTION-training_evaluation,model-Custom_Object_Detection_Gen3_ATSS,dataset-bbcd,num_iters-CONFIG,batch-CONFIG,usecase-reallife`
* `ACTION-nncf_export_evaluation,model-Custom_Image_Classification_EfficinetNet-B0,dataset-lg_chem,num_epochs-CONFIG,batch-CONFIG,usecase-reallife`

*TODO: insert example from expected metrics files*

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
Since the test suite works with OTE, each "test case" is considered as a situation that could be
happened during some process of work with OTE, and the process may include different actions.

Since OTE is focused on training a neural network and making some operations on the trained model,
we defined the test case by the parameters that define training process
(at least they defines it as much as it is possible for such stochastic process).

Usually the parameters defining the training process are:
1. a model - typically it is a name of OTE template to be used
2. a dataset - typically it is a dataset name that should be used
   (we use known pre-defined names for the datasets on our CI)
3. other training parameters:
   * `batch_size`
   * `num_training_epochs`

We suppose that for each algo backend there is a known set of parameters that define training
process, and we suppose that if two tests have the same these parameters, then they are belong to
the same test case.  
We call these parameters "the parameters defining the test case".

But from pytest point of view there are just a lot of tests with some parameters.

The general approach that is used to allow re-using results of test stages between test is the
following:
* The tests are grouped such that the tests from one group have the same parameters from the list
  of "parameters that define the test case" -- it means that the tests are grouped by the
  "test cases"
* After that the tests are reordered such that
  * the test from one group are executed sequentially one-by-one, without tests from other group
    between tests in one group
  * the test from one group are executed sequentially in the order defined for the test actions
    beforehand;
* An instance of the test case class is created once for each of the group of tests stated above
  -- so, the instance of test case class is created for each "test case" described above.  

As stated above, the instance of test case class is kept inside cache in OTE Test Helper class, it
allows to use the results of the previous tests of the same test case in the current test.

### V.2 Base interface of a test case class, creation of a test case class

The class of the test case is generated "on the fly" by the function
`generate_ote_integration_test_case_class` from the file `test_suite/training_test_case.py`;
the function has the declaration
```python
def generate_ote_integration_test_case_class(
    test_actions_classes: List[Type[BaseOTETestAction]],
) -> Type:
```
The function `generate_ote_integration_test_case_class` works as follows:
* receives as the input the list of action classes that should be used in the test case
  -- the test case will be a storage for the test stages wrapping the actions and will connect the
  test stages with each other
* and returns the class type that will be used by the instance of the test helper.

The variable with the type of test case received from the function is stored in the test helper
instance -- it is stored in a special class "test creation parameters", see about it below in the
section TODO.

Note that the result of this function is a `class`, not an `instance` of a class.  
Also note that the function receives list of action `classes`, not `instances` -- the instances of
test action classes are created when the instance of the test case class is created.

The class of the test case for a test is always inherited from the abstract interface class
`OTETestCaseInterface`.
It is derived from the abstract interface class `OTETestStagesStorageInterface`, so it has the
abstract method `get_stage` that for a string `name` returns test stage instance with this name.

The interface class `OTETestCaseInterface` has two own methods:
* abstract classmethod `get_list_of_test_stages` without parameters that returns the list of names
  of test stages for this test case
* abstract method `run_stage` that runs a stage with pointed name, the method has declaration
```python
    @abstractmethod
    def run_stage(self, stage_name: str, data_collector: DataCollector,
                  cur_test_expected_metrics_callback: Optional[Callable[[], Dict]]):
```

When the test case method `run_stage` is called, it receives as the parameters
* `stage_name` -- the name of the test stage to be called
* `data_collector` -- the `DataCollector` instance that is used when the method `run_once` of the
  test stage is called
* `cur_test_expected_metrics_callback` -- a factory function that returns the expected metrics for
  the current test, the factory function is used to create the `Validator` instance that will make
  validation for the current test.

The method `run_stage` of a created test case class always does the following:
1. checks that `stage_name` is a known name of a test stage for this test case
2. creates a `Validator` instance for the given `cur_test_expected_metrics_callback`
3. finds the test stage instance for the given `stage_name` and run for it `run_once` method as
   described above in the section "IV.2 Running a test action through its test stage" with the
   parameters `data_collector` and validator

If we return back to the `OTETestCaseInterface`, we can see that the test case class derived from it
should implement the classmethod `get_list_of_test_stages` without parameters that returns the list
of names of test stages for this test case.

Note that this method `get_list_of_test_stages` is a classmethod, since it is used when pytest
collects information on tests, before the first instance of the test case class is created.

> NB: We decided to make the test case class as a class that is generated by a function instead of a
> "normal" class, since we would like to encapsulate everything related to the test case in one
> entity -- due to it the method`get_list_of_test_stages` is not a method of a separate entity, but
> a classmethod of the test case class.  
> This could be changed in the future.

Also note that the function `generate_ote_integration_test_case_class` does not makes anything
really complicated for creation of a test case class: all test case classes are the same except the
parameter `test_actions_classes` with the list of action classes that is used to create test stage
wrapper for each of the test action from the list.

### V.3 The constructor of a test case class

As stated above, the function `generate_ote_integration_test_case_class` receives as a parameter
list of action `classes`, not `instances` -- the instances of test action classes are created when
the instance of the test case class is created.  
That is during construction of test case class its constructor creates instances of all the actions.

Each test case class created by the function `generate_ote_integration_test_case_class` has
the following constructor:
```python
def __init__(self, params_factories_for_test_actions: Dict[str, Callable[[], Dict]]):
```

The only parameter of this constructor is `params_factories_for_test_actions` that is
a dict:
* each key of the dict is a name of a test action
* each value of the dict is a factory function without parameters that returns the
  structure with kwargs for the constructor of the corresponding action

Note that most of the test actions do not receive parameters at all -- they receive the result of
previous actions, makes its own action, may make validation, etc.

For this case if the dict `params_factories_for_test_actions` does not contain as a key the name of
an action, then the constructor of the corresponding action will be called without parameters.

The constructor works as follows:
* For each action that was passed to the function `generate_ote_integration_test_case_class` that
  created this test case class
  * take name of the action
  * take `cur_params_factory = params_factories_for_test_actions.get(name)`
    * if the result is None, `cur_params = {}`
    * otherwise, `cur_params = cur_params_factory()`
  * call constructor of the current action as  
    `cur_action = action_cls(**cur_params)`
  * wraps the current action with the class `OTETestStage` as follows:  
    `cur_stage = OTETestStage(action=cur_action, stages_storage=self)`
  * store the current stage instance as  
    `self._stages[cur_name] = cur_stage`

## VI. Test Helper class

### VI.1 General description

Training test helper class `OTETestHelper` is implemented in `test_suite/training_tests_helper.py`.  
An instance of the class controls all execution of tests and keeps in its cache an instance of a
test case class between runs of different tests.

The most important method of the class are
* `get_list_of_tests` -- allows pytest trick generating test parameters for the test class.
  When pytest collects the info on all tests, the method returns structures that allows to make
  "pytest magic" to group and reorder the tests (see details below).
* `get_test_case` -- gets an instance of the test case class for the current test parameters, allows
  re-using the instance between several tests.

Note that the both of the methods work with test parameters that are used by pytest.

### VI.2 How pytest works with test parameters

Since `OTETestHelper` makes all the operations related to pytest parametrization mechanisms, we need
to describe here how pytest works with test parameters.

Generally pytest works as follows:
1. Pytest collects test info, for each test function or test method it gets information on
   parameters of the test and possible combination of parameters that may be executed.
2. Then pytest makes filtering -- it selects/deselects tests based on the pytest parameters
   (e.g. `-k`) and the names of the tests  
   -- each test with some combination of parameters has a full name of "test with parameters" that
   uniquely identifies the test with the parameters
2. Then pytest executes the selected tests one by one.
   When pytest executes a test function or a test method it gets a concrete combinations of
   parameter values for the parameters of the test and executes the test function/method with this
   combination.
   During the execution pytest may print the full name of the "test with parameters"

**How pytest gets information on parameters**

In pytest the information on test parameters for each test function/method consists of the following
3 elements:  
(NB: it is a short and may be approximate description! do not use it as a pytest documentation)
1. `argnames` -- a tuple of names of parameters of the test, typically this is a short tuple of
   strings
   * its length is the number of parameters of the test,
   * it contains string names of the parameters
2. `argvalues` -- a list of parameters of the test, this is a long list,
   * its length is the number of different combination of parameter values for the test,
   * each element of the list should be a tuple,
   * the length of each of the tuples is the same as the length of `argnames` above,
   * the tuple stores a concrete combination of values of the parameters
3. `ids` -- a list of string identifiers,
   * the list has the same length as the list `argvalues`
   * each value is a string
   * the string is used as an ID of the concrete combination of parameters  
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
`ote_pytest_generate_tests_insertion` below in the section TODO.

**How pytest runs a test with a combination of parameters**

When pytest runs a test function/method that has some parameters, pytest works as follows:  
(NB: it is a short and may be approximate description! do not use it as a pytest documentation)
1. gets the triplet `argnames, argvalues, ids` for this test function/method
2. check that the test function/method has all the parameters with names from the tuple `argnames`
2. makes filtering (selecting/deselecting) of concrete parameter values combinations as on pairs of
   `zip(argvalues, ids)` based on `ids` string identifiers and different pytest command line
   arguments (see pytest option `-k`)
3. for each selected combination of parameter values -- a pair `(arvalue_el, id)` from
   `zip(argvalues, ids)` -- do the following:
   * check that `argvalue_el` is a tuple with the length equal to `argnames`
   * create kwargs dict for the test function/method
   * sets in the kwargs dict for each key from `argnames` the corresponding value from
     `argvalue_el` probably in the following manner:  
     `for i in range(len(argnames)): kwargs[argnames[i]] = argvalue_el[i]`

### VI.3 How pytest parametrization mechanisms relates to the test suite and `OTETestHelper`

**(IMPORTANT)** The description how pytest works with test functions/methods parametrization in the
previous section relates to all pytest-based code.
But we would like to describe some important points related to `OTETestHelper` and the test suite as
a whole:

* typically for one OTE task type for all training tests there is only one test class with only only
  one test method that has a lot of combination of test parameters values
* the method `get_list_of_tests` of `OTETestHelper` returns this triplet
  `argnames, argvalues, ids` that is used later in `pytest_generate_tests`-related pytest magic to
  parametrize this test method  
  Note that the triplet `argnames, argvalues, ids` received from `get_list_of_tests` is used as is
  without any changes.
* `OTETestHelper` always defines `argnames = ("test_parameters",)`, so formally the only test method
  uses **only one** test parameter to parametrise tests, but values of the parameter are dict-s that
  contain info on real test parameters

### VI.4 Constructor of the class `OTETestHelper`

The constructor of the class `OTETestHelper` has the following declaration
```python
def __init__(self, test_creation_parameters: OTETestCreationParametersInterface):
```

As you can see it receives as the only parameter the class that is derived from
`OTETestCreationParametersInterface`.
We will refer to it as a *test parameters class* and we will refer to the base class
`OTETestCreationParametersInterface` as to *test parameters interface*.

We suppose that such test parameter class derived from `OTETestCreationParametersInterface` contains
most of information required to connect the test suite with a concrete algo backend.  
All the methods of the interface class are abstract methods without parameters that return
structures making this connection.

Example of such implementation is the class `DefaultOTETestCreationParametersInterface` that
contains implementation of almost all the test parameter class methods for mmdetection algo backend
(mmdetection is chosen due to historical reasons).  
Nevertheless, although these methods are implemented for mmdetection, most of them may
be used without modification (or with only slight modification) for other algo backends.

The constructor of the class `OTETestHelper` indeed makes the following:
* calls the methods of the received parameter class instance and stores the info received as
  the result of the calls in the `OTETestHelper` instance fields
* check that the info stored in `OTETestHelper` instance fields has a proper structure
* initialize a cache to store a test case class


### VI.5 Methods of the test case parameters class `OTETestCreationParametersInterface`

Let's consider all the methods of the abstract test parameters interface class one by one.

#### VI.5.1 `test_case_class`

```python
@abstractmethod
def test_case_class(self) -> Type[OTETestCaseInterface]:
```
 -- The method returns a class that will be used as a Test Case class for training tests.
Note that it should return a class itself (not an instance of the class).
Typically OTE Test Case class should be generated by the function
`training_test_case.generate_ote_integration_test_case_class`.
See details above in the section "V. Test Case class"

#### VI.5.2 `test_bunches`

```python
@abstractmethod
def test_bunches(self) -> List[Dict[str, Any]]:
```
-- The method returns a test bunches structure, it defines the combinations of test parameters for
which the test suite training test should be run.

This is the most important method since it defines the scope of the tests.

The method should return a list of dicts, each of the dicts defines one test case -- see description
how test cases are defined in the section "V.1 General description of test case class".  
All keys of the dicts are strings.

**(IMPORTANT)**
Note that in a typical situation a dict from the test bunches list is passed to the only test method
as the value `test_parameters` -- see "IMPORTANT" notice in the previous section
"VI.3 How pytest parametrization mechanisms relates to the test suite and `OTETestHelper`"


Mandatory keys of the dicts are:
* `"model_name"` -- the value is a string that is the name of a model to work with as it is defined
  in the template.yaml file of the model
* `"dataset_name"` -- the value is a string that is the name of the dataset, note that we use known
  pre-defined names for the datasets on our CI
* `"usecase"` -- the value is a string, if it is equal to `REALLIFE_USECASE_CONSTANT="reallife"`
  then validation will be run for the tests

Also typical non-mandatory keys of the dicts are
* `"num_training_iters"` or `"num_training_epochs"` or `"patience"` -- integer the parameter
  restricting the training time
* `"batch_size"` -- integer parameter, affects training speed and quality


Note that the following additional tricks are used:
1. For mandatory fields `"model_name"` and `"dataset_name"` the value may be not only a string, but
   a list of strings -- in this case a Cartesian product of all possible pairs `(model, dataset)` is
   used.  
   This is because this method is called `test_bunches` -- since each element of the returned list
   may define a "bunch" of tests
2. If a non-mandatory key in a test bunch dict equals to a string "DEFAULT", then it may be replaced
   by some default value pointed by the method `default_test_parameters` (see it below)

Note that also most of training actions (e.g. `OTETestTrainingAction`) use one more additional
trick: if values either for `batch_size` key or for `num_training_iters` key in a test bunch dict
contain a string constant `KEEP_CONFIG_FIELD_VALUE="CONFIG"` instead of an integer value, the action
reads the values of such parameters from the template file of the model or internal config of the
model and do not change them.  
It is important when we want to keep some training parameters "as is" and do not want to point our
own values for them.

Example of a test bunch
```
[
    dict(
        model_name=[
           'gen3_mobilenetV2_SSD',
           'gen3_mobilenetV2_ATSS',
           'gen3_resnet50_VFNet',
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

#### VI.5.3 `default_test_parameters`

```python
@abstractmethod
def default_test_parameters(self) -> Dict[str, Any]:
```
-- The method returns a dict that points for test parameters the default values.

If some dict in test bunches does not have a field that is pointed
in the dict returned by `default_test_parameters`, the value for the field is set by the default
value.

### VI.4 How the method `OTETestHelper.get_list_of_tests` works

The method `get_list_of_tests` of the class `OTETestHelper` works as follows:
* 




-------------------------------------------------------------

Usually pytest collects this information inside itself, but our test suite uses special interface
that allows to change it: if pytest finds the function `pytest_generate_tests` with declaration
```python
def pytest_generate_tests(metafunc):
```
then special pytest magic is allowed
: this function is called for each pytest function/method and
gives the following possibility through its parameter `metafunc`:
* `metafunc.cls` -- the type of the current test class of the test method (None for test functions)
* `metafunc.config.getoption("--some-option-name")` -- allows to get value of an additional pytest
  option declared in `pytest_addoption`
* 















-----------------------------------------------

  Note that the running of pytest tests for test suite is parametrized by some python magic
  that is implemented in the function `ote_pytest_generate_tests_insertion` in the file
  `test_suite/pytest_insertions.py`


* Instance of a test class
  Typically this class is defined in `test_ote_training.py` in the algo backend.
  This class contains some fixtures implementation.
  This class should be derived from the interface `OTETrainingTestInterface`.
