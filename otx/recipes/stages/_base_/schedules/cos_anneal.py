_base_ = "./schedule.py"

lr_config = dict(policy="CosineAnnealing", min_lr_ratio=0.0001)
