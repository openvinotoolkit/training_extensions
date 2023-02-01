_base_ = ["./yolox.py"]

model = dict(
    type="CustomYOLOX",
    bbox_head=dict(
        type="CustomYOLOXHead",
    ),
)

ignore = False
