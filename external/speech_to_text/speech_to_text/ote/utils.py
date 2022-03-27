# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.shapes.rectangle import Rectangle


def ote_extract_speech_to_text_dataset(dataset, annotation=True):
    """Convert OTE Speech To text datset to PyTorch format."""
    samples = []
    for sample in dataset:
        samples.append({
            "audio_path": sample.media.file_path,
            "text": sample.annotation_scene.annotations[0].get_labels()[0].name if annotation else ""
        })
    return samples


def ote_extract_eval_annotation(preds, tgts):
    """Convert OTE eval annotation to PyTorch lightning format."""
    annotation = {"pred": [], "tgt": []}
    for idx, (pred, tgt) in enumerate(zip(preds, tgts)):
        annotation["pred"].append(pred.annotation_scene.annotations[0].get_labels()[0].name)
        annotation["tgt"].append(tgt.annotation_scene.annotations[0].get_labels()[0].name)
    return annotation


def text_to_annotation(text: str) -> Annotation:
    """Convert plain text to Annotation format."""
    label = LabelEntity(name=text, domain=Domain.CLASSIFICATION)
    labels = [ScoredLabel(label)]
    annotation = Annotation(shape=Rectangle(x1=0, y1=0, x2=1, y2=1), labels=labels)
    return annotation
