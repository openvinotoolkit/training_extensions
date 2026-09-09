#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import os
from functools import cache
from importlib import resources
from typing import Any

import yaml

from app.models.model_manifest import ModelManifest
from app.supported_models import manifests
from app.supported_models.timm import TimmManifestProvider, id_to_model_name


class ManifestNotFoundException(Exception):
    """Exception raised when a model manifest is not found."""

    def __init__(self, model_manifest_id: str):
        super().__init__(f"Model architecture with ID '{model_manifest_id}' not found.")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested mappings are merged key by key. Any non-mapping value (including
    ``None``, lists and scalars) from ``override`` replaces the corresponding
    value in ``base`` wholesale.

    Args:
        base: The base mapping whose values may be overridden.
        override: The mapping whose values take precedence.

    Returns:
        dict[str, Any]: A new dictionary containing the merged result.
    """
    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


class ModelManifestService:
    """Service for loading and managing model manifests."""

    @staticmethod
    def _parse_manifest(*manifest_sources, relative: bool = True) -> ModelManifest:
        """
        Parse model manifest YAML files and merge them into an Algorithm object.

        This function takes multiple manifest source paths, merges them using hierarchical
        YAML configuration, and converts the merged result into a structured Algorithm model.

        Args:
            *manifest_sources: Variable length list of YAML manifest file paths.
                Files are merged in order, with later files overriding values from earlier files.
            relative: If True, the paths are treated as relative to the package where the
                manifest files are stored. If False, they are treated as absolute paths.

        Returns:
            ModelManifest: A populated Algorithm object containing the parsed manifest data.

        Note:
            Manifest files are merged with a recursive deep merge: nested mappings are
            combined key by key, while lists and scalar values from later files replace
            those from earlier files. Raises if any of the specified manifest files
            cannot be found.
        """
        if relative:
            manifest_sources = tuple(str(resources.files(manifests).joinpath(path)) for path in manifest_sources)
        merged_manifest: dict = {}
        for source in manifest_sources:
            with open(source, encoding="utf-8") as manifest_file:
                loaded: Any = yaml.safe_load(manifest_file)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Manifest '{source}' must be a YAML mapping at the top level.")
            merged_manifest = _deep_merge(merged_manifest, loaded)
        return ModelManifest(**merged_manifest)  # pyrefly: ignore[missing-argument,bad-unpacking]

    @classmethod
    def get_model_manifest_by_id(cls, model_manifest_id: str) -> ModelManifest:
        """
        Retrieve a model manifest by its ID, static or timm-derived.

        Looks up the ID among statically loaded YAML manifests first; if not
        found and the ID matches the timm-backed namespace, builds the manifest
        on the fly via `TimmManifestProvider`.

        Returns:
            ModelManifest: The ModelManifest object corresponding to the given ID.

        Raises:
            ManifestNotFoundException: If the model manifest with the given ID does not exist
                                    in the available model manifests.
        """
        static = cls.get_model_manifests()
        if model_manifest_id in static:
            return static[model_manifest_id]
        if TimmManifestProvider.is_timm_id(model_manifest_id):
            return TimmManifestProvider.build_manifest(id_to_model_name(model_manifest_id))
        raise ManifestNotFoundException(model_manifest_id=model_manifest_id)

    @classmethod
    @cache
    def get_model_manifests(cls) -> dict[str, ModelManifest]:
        """
        Find and load all model manifest files in the manifests directory.

        Scans the manifests directory hierarchy and loads YAML configuration files
        following a structured format where base configurations are inherited and
        overridden by more specific configurations.

        Returns:
            dict[str, ModelManifest]: A dictionary mapping model manifest IDs to their
            corresponding ModelManifest objects.

        Note:
            Directory structure follows:
            - manifests/base.yaml: Base configuration for all models
            - manifests/task/base.yaml: Base configuration for specific task types
            - manifests/task/model.yaml: Model-specific configurations
        """
        # Get the manifests directory path
        manifests_dir = resources.files(manifests)
        root_base_yaml = "base.yaml"

        # Find all model-specific YAML files
        model_manifests = {}

        # Iterate through task type directories
        for task_dir in [d for d in manifests_dir.iterdir() if d.is_dir()]:
            task_base_yaml = os.path.join(task_dir.name, "base.yaml")

            # Find all model files in this task directory
            for file in task_dir.iterdir():
                if file.name.endswith(".yaml") and file.name != "base.yaml":
                    model_yaml_path = os.path.join(task_dir.name, file.name)

                    # Build dependency chain: base -> task_base -> model
                    dependency_chain = [root_base_yaml, task_base_yaml, model_yaml_path]

                    # Parse manifest with all dependencies in order
                    manifest = cls._parse_manifest(*dependency_chain, relative=True)
                    model_manifests[manifest.id] = manifest

        return model_manifests
