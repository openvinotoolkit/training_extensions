_base_ = ["./dist/dist.py"]

cudnn_benchmark = True

seed = 5
deterministic = False

hparams = dict(dummy=0)

task_adapt = dict(op="REPLACE")
