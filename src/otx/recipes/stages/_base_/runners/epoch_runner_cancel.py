_base_ = "./epoch_runner.py"

runner = dict(type="EpochRunnerWithCancel", max_epochs=300)
