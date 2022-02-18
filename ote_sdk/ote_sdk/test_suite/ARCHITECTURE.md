h1. OTE SDK test suite architecture

h2. General description

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

h2. General architecture overview

Here and below we will write paths to test suite library files relatively with the folder
`ote_sdk/ote_sdk` of OTE git repository, so path to this file is referred as
`test_suite/ARCHITECTURE.md`.

When we run some test that uses `test_suite` library (typically `test_ote_training.py` in some of
the algo backends) the callstack of the test looks as follows:

* Pytest framework

* Instance of a test class
  Typically this class is defined in `test_ote_training.py` in the algo backend.
  This class contains some fixtures implementation and uses test helper (see the next item).
  The name of the class is started from `Test`, so pytest uses it as a usual test class.

* Instance of training test helper class `OTETestHelper` from `test_suite/training_tests_helper.py`.
  The instance of the class should be a static field of the test class stated above.
  The class controls all execution of tests.
  Also the class keeps in its cache an instance of test case class between runs of different tests
  (see the next item).

* Instance of a test case class.
  This class keeps in its fields results of all test stages between tests.
  Note that each test executes only one test stage.
  (Since the instance of this class is kept in cache of training test helper's instance between runs
  of tests, the results of one test may be re-used by other tests.)

* Instance of the test stage class `OTETestStage` from `test_suite/training_tests_stage.py`.
  The class wraps a test action class (see the next item).
  Also it makes validation of the results of the wrapped test action if it is required.

* Instance of a test action class
  The class makes the real actions that should be done for a test using calls of OTE SDK interfaces.

The next sections will describe the corresponding classes from the bottom to the top.


h2. Test actions.

h3. General description of test actions classes.

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

h3. When implementation of own test action class is required.

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

h3. How to implement own test action class.

Please, note that this section covers the topic how to implement a new test action class, but does
not cover the topic how to make the test action class to be used by tests -- it is covered below in
the section TODO <========================================================================================

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

h2. Test stage class

The class `OTETestStage` from `test_suite/training_tests_stage.py` works as a wrapper for a test
action. For each instance of a test action an instance of the class `OTETestStage` is created.

It's constructor has declaration
```python
def __init__(self, action: BaseOTETestAction, stages_storage: OTETestStagesStorageInterface):
```










-----------------------------------------------

  Note that the running of pytest tests for test suite is parametrized by some python magic
  that is implemented in the function `ote_pytest_generate_tests_insertion` in the file
  `test_suite/pytest_insertions.py`


* Instance of a test class
  Typically this class is defined in `test_ote_training.py` in the algo backend.
  This class contains some fixtures implementation.
  This class should be derived from the interface `OTETrainingTestInterface`.
