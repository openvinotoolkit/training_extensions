_base_ = "./log.py"

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
