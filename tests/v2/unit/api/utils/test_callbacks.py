"""OTX V2 API-utils Unit-Test codes (Callbacks)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: This test is copied from otx 1.X test. so we need to check again.

from time import time

from otx.v2.api.entities.train_parameters import UpdateProgressCallback, default_progress_callback
from otx.v2.api.utils.callbacks import Callback, TimeMonitorCallback
from pytest_mock.plugin import MockerFixture


class TestCallback:
    def test_callback_abstract_methods(self) -> None:
        """
        <b>Description:</b>
        Check "Callback" class object abstract methods

        <b>Input data:</b>
        "Callback" class object

        <b>Expected results:</b>
        Test passes if none of "Callback" class abstract methods raised an exception
        """
        callback = Callback()
        callback.on_epoch_begin(epoch=1, logs="on_epoch_begin logs")
        callback.on_epoch_end(epoch=1, logs="on_epoch_end logs")
        callback.on_batch_begin(batch=1, logs="on_batch_begin logs")
        callback.on_batch_end(batch=1, logs="on_batch_end logs")
        callback.on_train_begin(logs="on_train_begin logs")
        callback.on_train_end(batch=1, logs="on_train_end logs")
        callback.on_train_batch_begin(batch=1, logs="on_train_batch_begin logs")
        callback.on_train_batch_end(batch=1, logs="on_train_batch_end logs")
        callback.on_test_begin(logs="on_test_begin logs")
        callback.on_test_end(logs="on_epoch_begin logs")
        callback.on_test_batch_begin(batch=1, logger=None)
        callback.on_test_batch_end(batch=1, logger=None)

    def test_set_params(self) -> None:
        callback = Callback()
        callback.set_params({"param1": "value1", "param2": "value2"})
        assert callback.params == {"param1": "value1", "param2": "value2"}

    def test_set_model(self, mocker: MockerFixture) -> None:
        callback = Callback()
        model = mocker.MagicMock()
        callback.set_model(model=model)
        assert callback.model == model

class TestTimeMonitorCallback:
    @staticmethod
    def time_monitor_callback() -> TimeMonitorCallback:
        return TimeMonitorCallback(
            num_epoch=3,
            num_train_steps=4,
            num_val_steps=5,
            num_test_steps=6,
            update_progress_callback=default_progress_callback,
        )

    @staticmethod
    def check_current_step_start_step_time_attributes(
        callback: TimeMonitorCallback,
        expected_step: int,
        expected_step_time_before: float,
    ) -> None:
        assert callback.current_step == expected_step
        assert callback.start_step_time > expected_step_time_before

    def test_time_monitor_callback_initialization(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object initialization

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized "TimeMonitorCallback" class object are equal to expected

        <b>Steps</b>
        1. Check attributes of "TimeMonitorCallback" class object initialized with default optional parameters
        2. Check attributes of "TimeMonitorCallback" class object initialized with specified optional parameters
        """

        def check_time_monitor_callback_attributes(
            actual_time_monitor_callback: TimeMonitorCallback,
            expected_step_history: int,
            expected_epoch_history: int,
            expected_update_progress_callback: UpdateProgressCallback,
        ) -> None:
            assert actual_time_monitor_callback.total_epochs == 3
            assert actual_time_monitor_callback.train_steps == 4
            assert actual_time_monitor_callback.val_steps == 5
            assert actual_time_monitor_callback.test_steps == 6
            assert actual_time_monitor_callback.steps_per_epoch == 9  # train_steps + val_steps
            assert actual_time_monitor_callback.total_steps == 33  # steps_per_epoch * total_epochs + num_test_steps
            assert actual_time_monitor_callback.current_step == 0
            assert actual_time_monitor_callback.current_epoch == 0
            assert actual_time_monitor_callback.start_step_time  # calculated using time.time() method
            assert actual_time_monitor_callback.past_step_duration == []
            assert actual_time_monitor_callback.average_step == 0
            assert actual_time_monitor_callback.step_history == expected_step_history
            assert actual_time_monitor_callback.start_epoch_time  # calculated using time.time() method
            assert actual_time_monitor_callback.past_epoch_duration == []
            assert actual_time_monitor_callback.average_epoch == 0
            assert actual_time_monitor_callback.epoch_history == expected_epoch_history
            assert not actual_time_monitor_callback.is_training
            assert actual_time_monitor_callback.update_progress_callback == expected_update_progress_callback

        # Checking attributes of "TimeMonitorCallback" initialized with default optional parameters
        num_epoch = 3
        num_train_steps = 4
        num_val_steps = 5
        num_test_steps = 6

        time_monitor_callback = TimeMonitorCallback(
            num_epoch=num_epoch,
            num_train_steps=num_train_steps,
            num_val_steps=num_val_steps,
            num_test_steps=num_test_steps,
        )
        check_time_monitor_callback_attributes(
            actual_time_monitor_callback=time_monitor_callback,
            expected_epoch_history=5,
            expected_step_history=50,
            expected_update_progress_callback=default_progress_callback,
        )
        # Checking attributes of "TimeMonitorCallback" initialized with specified optional parameters
        step_history = 10
        epoch_history = 100
        callback = default_progress_callback
        time_monitor_callback = TimeMonitorCallback(
            num_epoch=num_epoch,
            num_train_steps=num_train_steps,
            num_val_steps=num_val_steps,
            num_test_steps=num_test_steps,
            step_history=step_history,
            epoch_history=epoch_history,
            update_progress_callback=callback,
        )
        check_time_monitor_callback_attributes(
            actual_time_monitor_callback=time_monitor_callback,
            expected_step_history=step_history,
            expected_epoch_history=epoch_history,
            expected_update_progress_callback=callback,
        )

    def test_time_monitor_callback_on_train_batch_begin(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_train_batch_begin" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "current_step" and "start_step_time" attributes of "TimeMonitorCallback" class object after
        "on_train_batch_begin" method are equal to expected
        """

        time_monitor_callback = self.time_monitor_callback()
        start_step_time_before = time() - 1
        time_monitor_callback.start_step_time = start_step_time_before
        time_monitor_callback.on_train_batch_begin(batch=1, logs="on_train_batch_begin logs")
        self.check_current_step_start_step_time_attributes(
            callback=time_monitor_callback,
            expected_step=1,
            expected_step_time_before=start_step_time_before,
        )
        time_monitor_callback.start_step_time = start_step_time_before
        time_monitor_callback.on_train_batch_begin(batch=2, logs="on_train_batch_begin logs")
        self.check_current_step_start_step_time_attributes(
            callback=time_monitor_callback,
            expected_step=2,
            expected_step_time_before=start_step_time_before,
        )

    def test_time_monitor_callback_on_train_batch_end(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_train_batch_end" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "past_step_duration" and "average_step" attributes of "TimeMonitorCallback" class object after
        "on_train_batch_end" method are equal to expected

        <b>Steps</b>
        1. Check "past_step_duration" and "average_step" attributes after "on_train_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_step_duration" attribute more than value of
        "step_history" attribute
        2. Check "past_step_duration" and "average_step" attributes after "on_train_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_step_duration" attribute less than value of
        "step_history" attribute
        """
        # Checking "past_step_duration" and "average_step" after "on_train_batch_end" for "TimeMonitorCallback" with
        # length of "past_step_duration" more than "step_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_step_duration = [10.0, 12.0]
        time_monitor_callback.step_history = 2
        time_monitor_callback.on_train_batch_end(batch=1, logs="on_train_batch_end logs")
        assert len(time_monitor_callback.past_step_duration) == 2
        assert round(time_monitor_callback.average_step, 4) == 6.0
        # Checking "past_step_duration" and "average_step" after "on_train_batch_end" for "TimeMonitorCallback" with
        # length of "past_step_duration" less than "step_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_step_duration = [10, 14]
        time_monitor_callback.step_history = 4
        time_monitor_callback.on_train_batch_end(batch=2, logs="on_train_batch_end logs")
        assert len(time_monitor_callback.past_step_duration) == 3
        assert round(time_monitor_callback.average_step, 4) == 8.0

    def test_time_monitor_callback_is_stalling(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "is_stalling" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if bool value returned by "is_stalling" method is equal to expected

        <b>Steps</b>
        1. Check value returned by "is_stalling" method for "TimeMonitorCallback" object with "is_training" attribute
        is "True" and "current_step" more than 2
        2. Check value returned by "is_stalling" method for "TimeMonitorCallback" object with "is_training" attribute
        is "True" and "current_step" equal to 2
        3. Check value returned by "is_stalling" method for "TimeMonitorCallback" object with "is_training" attribute
        is "False" and "current_step" more than 2
        """
        time_monitor_callback = self.time_monitor_callback()
        # Checking value returned by "is_stalling" for "TimeMonitorCallback" with "is_training" is "True" and
        # "current_step" more than 2
        time_monitor_callback.is_training = True
        time_monitor_callback.current_step = 3
        time_monitor_callback.start_step_time = 40
        assert time_monitor_callback.is_stalling()
        # Checking value returned by "is_stalling" for "TimeMonitorCallback" with "is_training" is "True" and
        # "current_step" equal to 2
        time_monitor_callback.current_step = 2
        assert not time_monitor_callback.is_stalling()
        # Checking value returned by "is_stalling" for "TimeMonitorCallback" with "is_training" is "False" and
        # "current_step" more than 2
        time_monitor_callback.current_step = 3
        time_monitor_callback.is_training = False
        assert not time_monitor_callback.is_stalling()

    def test_time_monitor_callback_on_test_batch_begin(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_test_batch_begin" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "current_step" and "start_step_time" attributes of "TimeMonitorCallback" class object after
        "on_test_batch_begin" method are equal to expected
        """
        time_monitor_callback = self.time_monitor_callback()
        start_step_time_before = time() - 1
        time_monitor_callback.start_step_time = start_step_time_before
        time_monitor_callback.on_test_batch_begin(batch=1, logger=None)
        self.check_current_step_start_step_time_attributes(time_monitor_callback, 1, start_step_time_before)
        time_monitor_callback.start_step_time = start_step_time_before
        time_monitor_callback.on_test_batch_begin(batch=2, logger=None)
        self.check_current_step_start_step_time_attributes(time_monitor_callback, 2, start_step_time_before)

    def test_time_monitor_callback_on_test_batch_end(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_test_batch_end" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "current_step" and "start_step_time" attributes of "TimeMonitorCallback" class object after
        "on_test_batch_end" method are equal to expected

        <b>Steps</b>
        1. Check "past_step_duration" and "average_step" attributes after "on_test_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_step_duration" attribute more than value of
        "step_history" attribute
        2. Check "past_step_duration" and "average_step" attributes after "on_test_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_step_duration" attribute less than value of
        "step_history" attribute
        """
        # Checking "past_step_duration" and "average_step" after "on_train_batch_end" for "TimeMonitorCallback" with
        # length of "past_step_duration" more than "step_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_step_duration = [5.0, 6.0]
        time_monitor_callback.step_history = 2
        time_monitor_callback.on_test_batch_end(batch=1, logger=None)
        assert len(time_monitor_callback.past_step_duration) == 2
        assert round(time_monitor_callback.average_step, 4) == 3.0
        # Checking "past_step_duration" and "average_step" after "on_train_batch_end" for "TimeMonitorCallback" with
        # length of "past_step_duration" less than "step_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_step_duration = [10.0, 14.0, 16.0]
        time_monitor_callback.step_history = 5
        time_monitor_callback.on_test_batch_end(batch=2, logger=None)
        assert len(time_monitor_callback.past_step_duration) == 4
        assert round(time_monitor_callback.average_step, 4) == 10.0

    def test_time_monitor_callback_on_train_begin(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_train_begin" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "is_training" attribute of "TimeMonitorCallback" class object after "on_train_begin" method is
        "True"

        <b>Steps</b>
        1. Check "is_training" attribute after "on_train_begin" method for "TimeMonitorCallback" class object with
        "is_training" attribute is "False"
        2. Check "is_training" attribute after "on_train_begin" method for "TimeMonitorCallback" class object with
        "is_training" attribute is "True"
        """
        # Checking "is_training" after "on_train_begin" for "TimeMonitorCallback" with "is_training" is "False"
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.on_train_begin()
        assert time_monitor_callback.is_training
        # Checking "is_training" after "on_train_begin" for "TimeMonitorCallback" with "is_training" is "True"
        time_monitor_callback.on_train_begin()
        assert time_monitor_callback.is_training

    def test_time_monitor_callback_on_train_end(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_train_end" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "current_step", "current_epoch" and "is_training" attributes of "TimeMonitorCallback" class
        object after "on_train_end" method are equal to expected

        <b>Steps</b>
        1. Check "current_step", "current_epoch" and "is_training" attributes after "on_train_end" method for
        "TimeMonitorCallback" class object with "is_training" attribute is "True"
        2. Check "is_training" attribute after "on_train_begin" method for "TimeMonitorCallback" class object with
        "is_training" attribute is "False"
        """

        def check_attributes_after_on_train_end(callback: TimeMonitorCallback) -> None:
            assert callback.current_step == 27  # total_steps - test_steps

            assert callback.current_epoch == 3  # total_epochs
            assert not callback.is_training

        # Checking "current_step", "current_epoch" and "is_training" after "on_train_end" for "TimeMonitorCallback" with
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.on_train_begin()  # setting "is_training" to "True"
        time_monitor_callback.on_train_end(batch=1)
        check_attributes_after_on_train_end(time_monitor_callback)
        # Checking "current_step", "current_epoch" and "is_training" after "on_train_end" for "TimeMonitorCallback" with
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.on_train_end(batch=1)
        check_attributes_after_on_train_end(time_monitor_callback)

    def test_time_monitor_callback_on_epoch_begin(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_epoch_begin" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "current_epoch" and "start_epoch_time" attributes of "TimeMonitorCallback" class object after
        "on_epoch_begin" method are equal to expected
        """
        start_epoch_time = time() - 1
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.start_epoch_time = start_epoch_time
        time_monitor_callback.on_epoch_begin(epoch=0, logs="on_epoch_begin logs")
        assert time_monitor_callback.current_epoch == 1
        assert time_monitor_callback.start_epoch_time > start_epoch_time
        time_monitor_callback.start_epoch_time = start_epoch_time
        time_monitor_callback.on_epoch_begin(epoch=2, logs="on_epoch_begin logs")
        assert time_monitor_callback.current_epoch == 3
        assert time_monitor_callback.start_epoch_time > start_epoch_time

    def test_time_monitor_callback_on_epoch_end(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "on_epoch_end" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if "past_epoch_duration" and "average_epoch" attributes of "TimeMonitorCallback" class object after
        "on_epoch_end" method are equal to expected

        <b>Steps</b>
        1. Check "past_epoch_duration" and "average_epoch" attributes after "on_test_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_epoch_duration" attribute more than value of
        "epoch_history" attribute
        2. Check "past_epoch_duration" and "average_epoch" attributes after "on_test_batch_end" method for
        "TimeMonitorCallback" class object with length of "past_epoch_duration" attribute less than value of
        "epoch_history" attribute
        """
        # Checking "past_epoch_duration" and "average_epoch" after "on_test_batch_end" for "TimeMonitorCallback" with
        # length of "past_epoch_duration" more than "epoch_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_epoch_duration = [4.0, 5.0, 10.0]
        time_monitor_callback.epoch_history = 3
        time_monitor_callback.on_epoch_end(epoch=1, logs="on_epoch_end logs")
        assert len(time_monitor_callback.past_epoch_duration) == 3
        assert round(time_monitor_callback.average_epoch, 4) == 5.0
        # Checking "past_epoch_duration" and "average_epoch" after "on_test_batch_end" for "TimeMonitorCallback" with
        # length of "past_epoch_duration" less than "epoch_history" value
        time_monitor_callback = self.time_monitor_callback()
        time_monitor_callback.past_epoch_duration = [4.0, 5.0]
        time_monitor_callback.epoch_history = 4
        time_monitor_callback.on_epoch_end(epoch=2, logs="on_epoch_end logs")
        assert len(time_monitor_callback.past_epoch_duration) == 3
        assert round(time_monitor_callback.average_epoch, 4) == 3.0

    def test_time_monitor_callback_get_progress(self) -> None:
        """
        <b>Description:</b>
        Check "TimeMonitorCallback" class object "get_progress" method

        <b>Input data:</b>
        "TimeMonitorCallback" class object with specified initialization parameters

        <b>Expected results:</b>
        Test passes if value returned by "get_progress" method is equal to expected
        """
        time_monitor_callback = self.time_monitor_callback()
        # Checking value returned by "get_progress" for "TimeMonitorCallback" with "current_step" equal to 0
        assert time_monitor_callback.get_progress() == 0.0
        # Checking value returned by "get_progress" for "TimeMonitorCallback" with "current_step" not equal to 0
        time_monitor_callback.current_step = 16
        time_monitor_callback.total_steps = 64
        assert time_monitor_callback.get_progress() == 25.0  # (current_step / total_steps)*100

    def test_getstate(self, mocker: MockerFixture) -> None:
        """
        Test the getstate method of the TimeMonitorCallback class.

        This method tests whether the getstate method of the TimeMonitorCallback class returns the expected state of the
        object. It does this by pickling the object and checking whether the pickled data is of type bytes.

        Args:
            mocker (pytest_mock.plugin.MockerFixture): A pytest-mock mocker fixture.

        Returns:
            None
        """
        import pickle
        time_monitor_callback = self.time_monitor_callback()
        with mocker.patch("otx.v2.api.utils.callbacks.dill.pickles", return_value=False):
            callback_data = pickle.dumps(time_monitor_callback)
            assert isinstance(callback_data, bytes)

    def test_deepcopy(self) -> None:
        """
        Test the deepcopy functionality of the time_monitor_callback object.
        """
        import copy
        time_monitor_callback = self.time_monitor_callback()
        callback_copy = copy.deepcopy(time_monitor_callback)
        assert time_monitor_callback.__dict__ == callback_copy.__dict__
