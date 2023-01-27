_base_ = ["./incremental.py"]

override_configs = dict(custom_hooks=[dict(type="TwoCropTransformHook", interval=5)])
