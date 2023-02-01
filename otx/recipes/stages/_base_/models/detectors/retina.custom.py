_base_ = ["./retina.py"]

model = dict(
    type="CustomRetinaNet",
    # pretrained=None,
    # backbone=dict(  # Replacding R50 by OTE MV2
    #    _delete_=True,
    #    type='mobilenetv2_w1',
    #    out_indices=(2, 3, 4, 5),
    #    frozen_stages=-1,
    #    norm_eval=True,  # False in OTE setting
    #    pretrained=True,
    # ),
    # neck=dict(
    #    in_channels=[24, 32, 96, 320],
    #    out_channels=64,
    # ),
    # bbox_head=dict(
    #    type='CustomRetinaHead',
    #    in_channels=64,
    #    feat_channels=64,
    # ),
)
