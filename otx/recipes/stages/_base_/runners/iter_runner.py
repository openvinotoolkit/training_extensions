_base_ = "./runner.py"

runner = dict(type="IterBasedRunner", max_iters=10000)

workflow = [("train", 1)]
