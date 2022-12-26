_base_ = './schedule.py'

lr_config = dict(
    policy='OneCycle',
    pct_start=0.2,
    div_factor=100,
    final_div_factor=1000
)
