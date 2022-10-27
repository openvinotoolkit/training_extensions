"""MobileNet-V3-Small for multi-class config."""

# pylint: disable=invalid-name

_base_ = "../base/models/mobilenet_v3.py"

model = dict(
    type="SAMImageClassifier",
    task="classification",
)
