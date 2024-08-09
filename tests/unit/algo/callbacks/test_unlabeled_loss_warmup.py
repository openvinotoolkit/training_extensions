import math

import pytest
from lightning import Trainer
from otx.algo.callbacks.unlabeled_loss_warmup import UnlabeledLossWarmUpCallback
from otx.algo.classification.efficientnet import EfficientNetForMulticlassCls


@pytest.fixture()
def fxt_semisl_model():
    return EfficientNetForMulticlassCls(10, train_type="SEMI_SUPERVISED")


def test_unlabeled_loss_warmup_callback(mocker, fxt_semisl_model):
    # Create a mock LightningModule and Trainer

    mock_trainer = mocker.MagicMock(spec=Trainer)
    mock_trainer.max_epochs = 10
    mock_trainer.model = fxt_semisl_model
    mock_trainer.train_dataloader = range(10)

    # Create an instance of UnlabeledLossWarmUpCallback
    callback = UnlabeledLossWarmUpCallback(warmup_steps_ratio=0.2)

    # Call the on_train_batch_start method and check the unlabeled_coef value
    callback.on_train_batch_start(mock_trainer, fxt_semisl_model, None, 0)
    assert math.isclose(callback.unlabeled_coef, 0.0)
    assert callback.total_steps == 20
    assert callback.current_step == 1

    # Call the on_train_batch_start method multiple times and check the unlabeled_coef value
    for i in range(1, 21):
        callback.on_train_batch_start(mock_trainer, fxt_semisl_model, None, i)
        expected_coef = 0.5 * (1 - math.cos(min(math.pi, (2 * math.pi * i) / 20)))
        assert math.isclose(callback.unlabeled_coef, expected_coef)

    # Call the on_train_batch_start method after reaching the total_steps and check the unlabeled_coef value
    callback.on_train_batch_start(mock_trainer, fxt_semisl_model, None, 11)
    assert math.isclose(callback.unlabeled_coef, 1.0)
