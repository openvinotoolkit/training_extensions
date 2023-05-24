_base_ = ["./incremental.py"]

custom_hooks = [dict(type="TwoCropTransformHook", interval=5)]
