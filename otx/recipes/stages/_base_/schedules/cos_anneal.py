_base_ = "./schedule.py"

lr_config = dict(policy="CosineAnnealing", warmup=None, warmup_iters=0, min_lr_ratio=0.00001)
