import pytest

from action_recognition.logging import TrainingLogger, PeriodicHandler, StreamHandler


class TestTrainingLogger:
    def test_dispatches_to_handler(self, mocker):
        logger = TrainingLogger()
        handler_mock = mocker.Mock()
        logger.register_handler('handler', handler_mock)
        logger.register_value('value', ['handler'])

        logger.log_value('value', 1)
        assert handler_mock.write.call_count == 1

    def test_raises_with_unknown_handler(self):
        logger = TrainingLogger()

        with pytest.raises(Exception):
            logger.register_value('value', ['handler'])

    def test_raises_with_unknown_value(self):
        logger = TrainingLogger()

        with pytest.raises(Exception):
            logger.log_value('value', 1)

    def test_dispatches_to_group_handler(self, mocker):
        logger = TrainingLogger()
        handler_mock = mocker.Mock()
        logger.register_handler('handler', handler_mock)

        logger.register_value_group(r'good_group/.*', ['handler'])

        logger.log_value('good_group/value', 1)
        assert handler_mock.write.call_count == 1

        with pytest.raises(Exception):
            logger.log_value('bad_group/value', 1)

    def test_dispatches_to_multiple_handlers(self, mocker):
        logger = TrainingLogger()
        handler1_mock = mocker.Mock()
        handler2_mock = mocker.Mock()
        logger.register_handler('handler1', handler1_mock)
        logger.register_handler('handler2', handler2_mock)
        logger.register_value('value', ['handler1', 'handler2'])

        logger.log_value('value', 1)
        assert handler1_mock.write.call_count == 1
        assert handler2_mock.write.call_count == 1

    def test_value_averaged_within_scope(self, mocker):
        logger = TrainingLogger()
        handler_mock = mocker.Mock()
        logger.register_handler('handler', handler_mock)
        logger.register_value('value', ['handler'], average=True)

        logger.log_value('value', 1)
        assert logger.get_value('value') == 1
        logger.log_value('value', 2)
        assert logger.get_value('value') == 1.5
        logger.log_value('value', 3)
        assert logger.get_value('value') == 2

    def test_saves_instant_values(self, mocker):
        logger = TrainingLogger()
        handler_mock = mocker.Mock()
        logger.register_handler('handler', handler_mock)
        logger.register_value('value', ['handler'], average=True)

        logger.log_value('value', 1)
        assert logger.get_value('value') == 1
        assert handler_mock.write.call_args_list[0][0][0].instant_value == 1
        logger.log_value('value', 2)
        assert logger.get_value('value') == 1.5
        assert handler_mock.write.call_args_list[1][0][0].instant_value == 2
        logger.log_value('value', 3)
        assert logger.get_value('value') == 2
        assert handler_mock.write.call_args_list[2][0][0].instant_value == 3


@pytest.fixture
def periodic_handler_mock(mocker):
    periodic_handler = PeriodicHandler(scope='test_scope')
    periodic_handler.flush = mocker.Mock()
    return periodic_handler


class TestPeriodicHandler:
    def test_value_resets(self, periodic_handler_mock):
        logger = TrainingLogger()
        logger.register_handler('handler', periodic_handler_mock)
        logger.register_value('scope/value', ['handler'], average=True)

        with logger.scope(scope='test_scope'):
            logger.log_value('scope/value', 1)
            logger.log_value('scope/value', 2)
            logger.log_value('scope/value', 3)
        assert logger.get_value('scope/value') == 2

        logger.reset_values('scope')

        with logger.scope(scope='test_scope'):
            logger.log_value('scope/value', 5)
            logger.log_value('scope/value', 10)
            logger.log_value('scope/value', 15)
        assert logger.get_value('scope/value') == 10

    def test_correct_scope_handled(self, periodic_handler_mock):
        logger = TrainingLogger()
        logger.register_handler('handler', periodic_handler_mock)
        logger.register_value('value', ['handler'])

        with logger.scope(scope='wrong_scope'):
            pass

        periodic_handler_mock.flush.assert_not_called()

        with logger.scope(scope='test_scope'):
            pass

        assert periodic_handler_mock.flush.call_count == 1


class TestStreamHandler:
    def test_flush_writes_to_stream(self, mocker):
        logger = TrainingLogger()
        stream = mocker.Mock()
        stream_handler = StreamHandler('epoch', stream=stream, fmt='* {values}')
        logger.register_handler('handler', stream_handler)
        logger.register_value('value', ['handler'])

        with logger.scope():
            logger.log_value('value', 1)
            logger.log_value('value', 0.1)

        stream.write.assert_called_once_with('* value 0.1000\n')
        assert stream.flush.call_count == 1

    def test_instant_values_displayed(self, mocker):
        logger = TrainingLogger()
        stream = mocker.Mock()
        stream_handler = StreamHandler('batch', stream=stream, fmt='* {values}')
        logger.register_handler('handler', stream_handler)
        logger.register_value('value', ['handler'], average=True)

        for i, x in logger.scope_enumerate(range(1, 4)):
            logger.log_value('value', x)

        assert stream.write.call_count == 3
        stream.write.assert_has_calls([
            mocker.call('* value 1.0000 (1.0000)\n'),
            mocker.call('* value 2.0000 (1.5000)\n'),
            mocker.call('* value 3.0000 (2.0000)\n'),
        ])
