_base_ = "../model.py"

task = "segmentation"

model = dict(train_cfg=dict(mix_loss=dict(enable=False, weight=0.1)), test_cfg=dict(mode="whole"))
