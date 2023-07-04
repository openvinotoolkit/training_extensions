_base_ = "./runner.py"

runner = dict(type="EpochBasedRunner", max_epochs=1)

workflow = [("train", 1)]
