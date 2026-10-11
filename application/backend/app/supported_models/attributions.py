# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Attribution of the original (upstream) pretrained models supported by Geti."""

from fnmatch import fnmatchcase
from functools import cache
from importlib import resources

import yaml
from pydantic import BaseModel, ConfigDict, Field

_ATTRIBUTIONS_RESOURCE = "attributions.yaml"


class ModelAttribution(BaseModel):
    """Attribution of the original work that a model architecture is derived from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Name of the original work")
    models: tuple[str, ...] = Field(description="fnmatch patterns of the model architecture IDs it applies to")
    creators: str = Field(description="Identification of the creators / licensors of the original work")
    copyright: str | None = Field(description="Verbatim upstream copyright notice, if any")
    source: str = Field(description="URL of the original work")

    def matches(self, model_architecture_id: str) -> bool:
        """Whether this attribution applies to the given model architecture ID."""
        return any(fnmatchcase(model_architecture_id, pattern) for pattern in self.models)


@cache
def list_model_attributions() -> tuple[ModelAttribution, ...]:
    """Load all the model attributions from `app/supported_models/licenses/attributions.yaml`."""
    raw = (
        resources.files("app.supported_models").joinpath("licenses", _ATTRIBUTIONS_RESOURCE).read_text(encoding="utf-8")
    )
    return tuple(ModelAttribution.model_validate(entry) for entry in yaml.safe_load(raw) or [])


def get_model_attribution(model_architecture_id: str) -> ModelAttribution | None:
    """Get the attribution of the original work a model architecture is derived from.

    Args:
        model_architecture_id: ID of the model architecture (model manifest ID).

    Returns:
        The first matching attribution, or None if no attribution is registered for the model.
    """
    return next((a for a in list_model_attributions() if a.matches(model_architecture_id)), None)
