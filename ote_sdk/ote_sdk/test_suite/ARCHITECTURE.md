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

We suppose that each test executes one test stage (also called test action)

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


## III. Test actions.

### III.1 General description of test actions classes.

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

*Example:*
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

### III.2 When implementation of own test action class is required.

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

### III.3 How to implement own test action class.

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


### IV.3 Validation of action results

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
criteria may be
* "the quality metric of the current action is not worse than the result of _that_ action with
  possible quality drop 1%"
* "the quality metric of the current action is the same as the result of _that_ action with
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

As we stated above the `validator is not None` is the necessary condition to run validation, but it
is not sufficient.  
The list of sufficient conditions to run real validation in `run_once` is as follows:
* The parameter `validator` of `run_once` method  satisfies `validator is not None`
  (i.e. the validation is run not from the dependency chain).
* For the action the field `_with_validation == True`.
  If `_with_validation == False` it means that validation for this action is impossible -- e.g.
  "export" action cannot be validated since it does not return quality metrics, but the action
  "evaluation after export" is validated.
* The current test has the parameter `usecase == "reallife"`
  If a test is not a "reallife" test it means that a real training is not made for the test,
  so we cannot expect real quality, so validation is not done.

To investigate in details the conditions see the declaration of constructor of the `Validator`
class:
```python
    def __init__(self, cur_test_expected_metrics_callback: Optional[Callable[[], Dict]]):
```
As you can see it receives only one parameter, and this parameter is NOT a structure that
describes the requirements for the expected metrics for the action, but the parameter is
a FACTORY that returns the structure.

It is required since
a. constructing the structure requires complicated operations and reading of YAML files,
b. if validation should be done for the current test, and the expected metrics for the tests are
   absent, the test MUST fail  
   (it is important to avoid situations when developers forget adds info on expected metrics and due
   to it tests are not failed)
c. but if validation for the current test is not required the test should not try to get the
   expected metrics

So to avoid checking of expected metrics structures for the tests without validation, an algo
backend a factory is used -- the factory for an action's validator is called if and only if
the action should be validated.

The factory is implemented in the test suite as a pytest fixture -- see the fixture
`cur_test_expected_metrics_callback_fx` in the file `test_suite/fixtures.py`.

The fixture works as follows:
* receives from other fixtures contents of the YAML file that is pointed to pytest as the pytest
  parameter `--expected-metrics-file`
* checks if the current test is "reallife" training or not,
* if it is not reallife then validation is not required -- in this case
  * the fixture returns None,
  * the Validator class receives None as the constructor's parameter instead of a factory,
  * Validator understands it as "skip validation"
* if this is reallife training test, the fixture returns a factory function

The returned factory function extracts from all expected metrics the expected metrics for the
current test (and if the metrics are absent -- fail the current test).











-----------------------------------------------

  Note that the running of pytest tests for test suite is parametrized by some python magic
  that is implemented in the function `ote_pytest_generate_tests_insertion` in the file
  `test_suite/pytest_insertions.py`


* Instance of a test class
  Typically this class is defined in `test_ote_training.py` in the algo backend.
  This class contains some fixtures implementation.
  This class should be derived from the interface `OTETrainingTestInterface`.
